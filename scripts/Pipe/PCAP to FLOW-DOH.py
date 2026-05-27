#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extrator de features DoH a partir de PCAP/PCAPNG.

Correções principais:
1) Suporte a Ethernet, Linux cooked capture v1/v2 e RAW IP.
2) packet_length corrigido:
   - IPv4: usa ip.len
   - IPv6: usa ip.plen + 40
   - fallback: len(buf)
3) Suporte a dois modos de amostragem:
   - flow: uma amostra por fluxo TCP/443
   - window: uma amostra por janela temporal dentro do fluxo
4) Geração do dataset_DoH.csv com as features DoH + label.
5) Label extraído do nome do arquivo:
   - DoH_Exfiltration_*.pcap ou .pcapng => Exfiltration
   - DoH_Benign_*.pcap ou .pcapng => Benign
6) Extração de SNI, ALPN e JA3 a partir do TLS ClientHello.
7) Preenchimento controlado de valores ausentes.
"""

import os
import re
import socket
import hashlib
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import dpkt


# ============================================================
# CONFIGURAÇÃO
# ============================================================

pasta_pcap = r"C:\Users\cborges\OneDrive - Fortinet\Documents\Studies\Doctorade UFU\Research\Development\Dataset\My Dataset\PCAPs\DoH"

pasta_saida = r"C:\Users\cborges\OneDrive - Fortinet\Documents\Studies\Doctorade UFU\Research\Development\Dataset\My Dataset\CSVs\DoH\Versao_Final"

os.makedirs(pasta_saida, exist_ok=True)

OUT_DOH = os.path.join(pasta_saida, "dataset_DoH.csv")
DOH_REPORT = os.path.join(pasta_saida, "doh_clienthello_report.csv")
STATS_DOH = os.path.join(pasta_saida, "stats_DoH.csv")

# Modo de amostragem:
# - "flow": gera 1 amostra por fluxo TCP/443, separando apenas por timeout de inatividade.
# - "window": gera amostras por janela temporal dentro de cada conexão TCP/443.
# Para gerar mais amostras, use "window".
SAMPLE_MODE = "window"

FLOW_TIMEOUT_SECONDS = 120.0

# Usado somente quando SAMPLE_MODE = "window".
WINDOW_SECONDS = 5.0

# Preenchimento de valores ausentes:
# - métricas numéricas ausentes recebem 0.0;
# - SNI, ALPN e JA3 ausentes recebem "unknown".
FILL_MISSING_VALUES = True
UNKNOWN_TLS_VALUE = "unknown"

DOH_FEATURES = [
    "src2dst_min_length",
    "src2dst_max_length",
    "src2dst_mean_length",
    "src2dst_stddev_length",

    "dst2src_min_length",
    "dst2src_max_length",
    "dst2src_mean_length",
    "dst2src_stddev_length",

    "bidirectional_max_length",
    "bidirectional_mean_length",
    "bidirectional_stddev_length",

    "src2dst_min_time",
    "src2dst_max_time",
    "src2dst_mean_time",
    "src2dst_stddev_time",

    "dst2src_min_time",
    "dst2src_max_time",
    "dst2src_mean_time",
    "dst2src_stddev_time",

    "bidirectional_min_time",
    "bidirectional_max_time",
    "bidirectional_mean_time",
    "bidirectional_stddev_time",

    "num_src2dst",
    "num_dst2src",
    "flow_duration",
    "byte_count",
    "handshake_time",

    "sni",
    "alpn",
    "ja3",
]

FINAL_COLUMNS = DOH_FEATURES + ["label"]


# ============================================================
# AUXILIARES
# ============================================================

def normalize_filename(nome_arquivo: str) -> str:
    nome = nome_arquivo.lower()
    nome = unicodedata.normalize("NFKD", nome)
    nome = nome.encode("ascii", "ignore").decode("ascii")
    return nome


def detect_label(nome_arquivo: str) -> str:
    nome = normalize_filename(nome_arquivo)

    tokens = re.split(r"[^a-z0-9]+", nome)
    tokens = [t for t in tokens if t]

    exfil_tokens = {
        "exfil",
        "exfiltration",
        "exfiltracao",
        "exfiltrac",
        "dataexfiltration",
        "tunnel",
        "tunneling",
        "dnscat",
        "dnscat2",
        "iodine",
        "c2",
        "malicious",
        "attack",
        "malware",
        "leak",
        "leakage",
    }

    benign_tokens = {
        "benign",
        "benigno",
        "normal",
        "legit",
        "legitimate",
    }

    if any(t in exfil_tokens for t in tokens):
        return "Exfiltration"

    if any(t in benign_tokens for t in tokens):
        return "Benign"

    if re.search(
        r"(exfil|exfiltration|exfiltrac|exfiltracao|dnscat|iodine|malware|attack|tunnel|c2)",
        nome,
    ):
        return "Exfiltration"

    return "Benign"


def is_doh_file(nome_arquivo: str) -> bool:
    nome = normalize_filename(nome_arquivo)

    if "doh" in nome:
        return True

    if "_443" in nome or "-443" in nome or "443" in nome:
        return True

    return False


def open_pcap_reader(fp, path):
    if path.lower().endswith(".pcapng"):
        return dpkt.pcapng.Reader(fp)

    return dpkt.pcap.Reader(fp)


def get_reader_datalink(reader):
    try:
        return reader.datalink()
    except Exception:
        pass

    try:
        return reader.datalink
    except Exception:
        pass

    return None


def decode_ip_packet(buf: bytes, datalink):
    """
    Suporta:
    - Ethernet: DLT 1
    - Linux cooked capture v1: DLT 113
    - Linux cooked capture v2: DLT 276
    - RAW IPv4/IPv6
    """

    if datalink == 1:
        try:
            eth = dpkt.ethernet.Ethernet(buf)
            if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                return eth.data
        except Exception:
            return None

    if datalink == 113:
        try:
            sll = dpkt.sll.SLL(buf)
            if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                return sll.data
        except Exception:
            return None

    if datalink == 276:
        try:
            if len(buf) < 20:
                return None

            protocol = int.from_bytes(buf[0:2], "big")
            payload = buf[20:]

            if protocol == 0x0800:
                return dpkt.ip.IP(payload)

            if protocol == 0x86DD:
                return dpkt.ip6.IP6(payload)

            return None
        except Exception:
            return None

    try:
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return eth.data
    except Exception:
        pass

    try:
        sll = dpkt.sll.SLL(buf)
        if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return sll.data
    except Exception:
        pass

    try:
        if len(buf) > 0:
            version = buf[0] >> 4

            if version == 4:
                return dpkt.ip.IP(buf)

            if version == 6:
                return dpkt.ip6.IP6(buf)
    except Exception:
        pass

    return None


def get_ip_addresses(ip):
    if isinstance(ip, dpkt.ip.IP):
        return socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst)

    if isinstance(ip, dpkt.ip6.IP6):
        return (
            socket.inet_ntop(socket.AF_INET6, ip.src),
            socket.inet_ntop(socket.AF_INET6, ip.dst),
        )

    return None, None


def get_packet_length(buf: bytes, ip) -> int:
    try:
        if isinstance(ip, dpkt.ip.IP):
            return int(ip.len)

        if isinstance(ip, dpkt.ip6.IP6):
            return int(ip.plen) + 40
    except Exception:
        pass

    return int(len(buf))


def safe_min(values):
    return float(min(values)) if values else np.nan


def safe_max(values):
    return float(max(values)) if values else np.nan


def safe_mean(values):
    return float(np.mean(values)) if values else np.nan


def safe_std(values):
    return float(np.std(values, ddof=0)) if values else np.nan


def inter_arrival_times(timestamps):
    if len(timestamps) <= 1:
        return []

    ts = sorted(timestamps)

    return [
        float((ts[i] - ts[i - 1]) * 1000.0)
        for i in range(1, len(ts))
    ]


def bidirectional_inter_arrival_times(src_ts, dst_ts):
    events = []

    for t in src_ts:
        events.append((float(t), "src2dst"))

    for t in dst_ts:
        events.append((float(t), "dst2src"))

    events.sort(key=lambda x: x[0])

    if len(events) <= 1:
        return []

    return [
        float((events[i][0] - events[i - 1][0]) * 1000.0)
        for i in range(1, len(events))
    ]


# ============================================================
# PARSER TLS CLIENTHELLO
# ============================================================

GREASE_VALUES = {
    0x0A0A, 0x1A1A, 0x2A2A, 0x3A3A,
    0x4A4A, 0x5A5A, 0x6A6A, 0x7A7A,
    0x8A8A, 0x9A9A, 0xAAAA, 0xBABA,
    0xCACA, 0xDADA, 0xEAEA, 0xFAFA,
}


def is_grease(value: int) -> bool:
    return value in GREASE_VALUES


def looks_like_clienthello_packet(tcp_data: bytes) -> bool:
    if not tcp_data or len(tcp_data) < 6:
        return False

    if tcp_data[0] != 22:
        return False

    if tcp_data[1] != 0x03:
        return False

    if tcp_data[5] != 1:
        return False

    return True


def try_parse_client_hello(buf: bytes):
    if not buf or len(buf) < 5:
        return None

    if buf[0] != 22:
        return None

    record_length = int.from_bytes(buf[3:5], "big")

    if len(buf) < 5 + record_length:
        return None

    record = buf[5:5 + record_length]

    if len(record) < 4:
        return None

    if record[0] != 1:
        return None

    handshake_length = int.from_bytes(record[1:4], "big")

    if len(record) < 4 + handshake_length:
        return None

    client_hello = record[4:4 + handshake_length]

    p = 0

    if len(client_hello) < 2 + 32 + 1:
        return None

    ja3_version = int.from_bytes(client_hello[p:p + 2], "big")
    p += 2

    p += 32

    session_id_len = client_hello[p]
    p += 1

    if len(client_hello) < p + session_id_len + 2:
        return None

    p += session_id_len

    cipher_suites_len = int.from_bytes(client_hello[p:p + 2], "big")
    p += 2

    if len(client_hello) < p + cipher_suites_len + 1:
        return None

    ciphers = []

    for i in range(0, cipher_suites_len, 2):
        value = int.from_bytes(client_hello[p + i:p + i + 2], "big")

        if not is_grease(value):
            ciphers.append(value)

    p += cipher_suites_len

    compression_methods_len = client_hello[p]
    p += 1

    if len(client_hello) < p + compression_methods_len + 2:
        return None

    p += compression_methods_len

    extensions_total_len = int.from_bytes(client_hello[p:p + 2], "big")
    p += 2

    if len(client_hello) < p + extensions_total_len:
        return None

    ext_end = p + extensions_total_len

    extensions = []
    supported_groups = []
    ec_point_formats = []

    sni = np.nan
    alpn = np.nan

    while p + 4 <= ext_end:
        ext_type = int.from_bytes(client_hello[p:p + 2], "big")
        p += 2

        ext_len = int.from_bytes(client_hello[p:p + 2], "big")
        p += 2

        if p + ext_len > ext_end:
            break

        ext_data = client_hello[p:p + ext_len]
        p += ext_len

        if not is_grease(ext_type):
            extensions.append(ext_type)

        if ext_type == 0 and ext_len >= 5:
            try:
                server_name_list_len = int.from_bytes(ext_data[0:2], "big")

                if 2 + server_name_list_len <= len(ext_data):
                    name_type = ext_data[2]
                    name_len = int.from_bytes(ext_data[3:5], "big")

                    if name_type == 0 and 5 + name_len <= len(ext_data):
                        sni_value = (
                            ext_data[5:5 + name_len]
                            .decode(errors="ignore")
                            .strip()
                        )

                        if sni_value:
                            sni = sni_value
            except Exception:
                pass

        if ext_type == 16 and ext_len >= 3:
            try:
                alpn_list_len = int.from_bytes(ext_data[0:2], "big")
                q = 2
                protocols = []

                while q < 2 + alpn_list_len and q < len(ext_data):
                    proto_len = ext_data[q]
                    q += 1

                    if q + proto_len <= len(ext_data):
                        proto = ext_data[q:q + proto_len].decode(errors="ignore")

                        if proto:
                            protocols.append(proto)

                    q += proto_len

                if protocols:
                    alpn = ",".join(protocols)
            except Exception:
                pass

        if ext_type == 10 and ext_len >= 2:
            try:
                groups_len = int.from_bytes(ext_data[0:2], "big")
                q = 2

                while q + 2 <= 2 + groups_len and q + 2 <= len(ext_data):
                    group_value = int.from_bytes(ext_data[q:q + 2], "big")
                    q += 2

                    if not is_grease(group_value):
                        supported_groups.append(group_value)
            except Exception:
                pass

        if ext_type == 11 and ext_len >= 1:
            try:
                point_len = ext_data[0]
                q = 1

                while q < 1 + point_len and q < len(ext_data):
                    point_value = ext_data[q]
                    q += 1
                    ec_point_formats.append(point_value)
            except Exception:
                pass

    ja3_ciphers = "-".join(str(x) for x in ciphers)
    ja3_extensions = "-".join(str(x) for x in extensions)
    ja3_groups = "-".join(str(x) for x in supported_groups)
    ja3_points = "-".join(str(x) for x in ec_point_formats)

    ja3_string = f"{ja3_version},{ja3_ciphers},{ja3_extensions},{ja3_groups},{ja3_points}"
    ja3_hash = hashlib.md5(ja3_string.encode("utf-8")).hexdigest()

    return {
        "sni": sni,
        "alpn": alpn,
        "ja3": ja3_hash,
        "ja3_string": ja3_string,
    }


# ============================================================
# PRÉ-SCAN TLS
# ============================================================

def build_tls_index_for_doh(pcap_path: str, max_buffer_per_flow: int = 65536):
    buffers = {}
    parsed = {}
    clienthello_packets_seen = 0
    tcp443_packets_seen = 0

    with open(pcap_path, "rb") as f:
        reader = open_pcap_reader(f, pcap_path)
        datalink = get_reader_datalink(reader)

        print(f"  Datalink do PCAP: {datalink}")

        for ts, buf in reader:
            try:
                ip = decode_ip_packet(buf, datalink)

                if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    continue

                tcp = ip.data

                if not isinstance(tcp, dpkt.tcp.TCP):
                    continue

                src_ip, dst_ip = get_ip_addresses(ip)

                if not src_ip or not dst_ip:
                    continue

                src_port = int(tcp.sport)
                dst_port = int(tcp.dport)

                if src_port == 443 or dst_port == 443:
                    tcp443_packets_seen += 1

                if dst_port != 443:
                    continue

                if not tcp.data:
                    continue

                if looks_like_clienthello_packet(tcp.data):
                    clienthello_packets_seen += 1

                base_key = (src_ip, dst_ip, src_port, 443)

                if base_key in parsed:
                    continue

                previous = buffers.get(base_key, b"")

                if len(previous) < max_buffer_per_flow:
                    previous += tcp.data
                    buffers[base_key] = previous

                info = try_parse_client_hello(buffers[base_key])

                if info:
                    parsed[base_key] = info
                    buffers.pop(base_key, None)

            except Exception:
                continue

    print(f"  Pacotes TCP/443 vistos no pré-scan TLS: {tcp443_packets_seen}")
    print(f"  ClientHello vistos no pré-scan TLS: {clienthello_packets_seen}")
    print(f"  Entradas TLS indexadas: {len(parsed)}")

    return parsed, clienthello_packets_seen, tcp443_packets_seen


# ============================================================
# AGREGAÇÃO
# ============================================================

def make_flow_record():
    return {
        "src2dst_lengths": [],
        "dst2src_lengths": [],
        "src2dst_timestamps": [],
        "dst2src_timestamps": [],
        "first_ts": None,
        "last_ts": None,
        "base_key": None,
    }


def append_packet_to_flow(flow, direction, packet_length, ts):
    if flow["first_ts"] is None:
        flow["first_ts"] = float(ts)

    flow["last_ts"] = float(ts)

    if direction == "src2dst":
        flow["src2dst_lengths"].append(int(packet_length))
        flow["src2dst_timestamps"].append(float(ts))
    else:
        flow["dst2src_lengths"].append(int(packet_length))
        flow["dst2src_timestamps"].append(float(ts))


def finalize_flow_to_row(flow, tls_index):
    src2dst_lengths = flow["src2dst_lengths"]
    dst2src_lengths = flow["dst2src_lengths"]

    src2dst_timestamps = flow["src2dst_timestamps"]
    dst2src_timestamps = flow["dst2src_timestamps"]

    all_lengths = src2dst_lengths + dst2src_lengths
    all_timestamps = src2dst_timestamps + dst2src_timestamps

    src2dst_iat = inter_arrival_times(src2dst_timestamps)
    dst2src_iat = inter_arrival_times(dst2src_timestamps)
    bidirectional_iat = bidirectional_inter_arrival_times(
        src2dst_timestamps,
        dst2src_timestamps,
    )

    if all_timestamps:
        flow_duration = float((max(all_timestamps) - min(all_timestamps)) * 1000.0)
    else:
        flow_duration = np.nan

    if src2dst_timestamps and dst2src_timestamps:
        handshake_time = float(
            (min(dst2src_timestamps) - min(src2dst_timestamps)) * 1000.0
        )
    else:
        handshake_time = np.nan

    byte_count = float(sum(all_lengths)) if all_lengths else 0.0

    base_key = flow["base_key"]
    tls_info = tls_index.get(base_key, {})

    return {
        "src2dst_min_length": safe_min(src2dst_lengths),
        "src2dst_max_length": safe_max(src2dst_lengths),
        "src2dst_mean_length": safe_mean(src2dst_lengths),
        "src2dst_stddev_length": safe_std(src2dst_lengths),

        "dst2src_min_length": safe_min(dst2src_lengths),
        "dst2src_max_length": safe_max(dst2src_lengths),
        "dst2src_mean_length": safe_mean(dst2src_lengths),
        "dst2src_stddev_length": safe_std(dst2src_lengths),

        "bidirectional_max_length": safe_max(all_lengths),
        "bidirectional_mean_length": safe_mean(all_lengths),
        "bidirectional_stddev_length": safe_std(all_lengths),

        "src2dst_min_time": safe_min(src2dst_iat),
        "src2dst_max_time": safe_max(src2dst_iat),
        "src2dst_mean_time": safe_mean(src2dst_iat),
        "src2dst_stddev_time": safe_std(src2dst_iat),

        "dst2src_min_time": safe_min(dst2src_iat),
        "dst2src_max_time": safe_max(dst2src_iat),
        "dst2src_mean_time": safe_mean(dst2src_iat),
        "dst2src_stddev_time": safe_std(dst2src_iat),

        "bidirectional_min_time": safe_min(bidirectional_iat),
        "bidirectional_max_time": safe_max(bidirectional_iat),
        "bidirectional_mean_time": safe_mean(bidirectional_iat),
        "bidirectional_stddev_time": safe_std(bidirectional_iat),

        "num_src2dst": int(len(src2dst_lengths)),
        "num_dst2src": int(len(dst2src_lengths)),
        "flow_duration": flow_duration,
        "byte_count": byte_count,
        "handshake_time": handshake_time,

        "sni": tls_info.get("sni", np.nan),
        "alpn": tls_info.get("alpn", np.nan),
        "ja3": tls_info.get("ja3", np.nan),
    }


def extract_doh_features_from_pcap(pcap_path: str, tls_index: dict) -> pd.DataFrame:
    """
    Extrai features DoH.

    SAMPLE_MODE = "flow":
        Gera uma amostra por fluxo TCP/443.

    SAMPLE_MODE = "window":
        Gera uma amostra por janela temporal dentro da mesma conexão TCP/443.
    """

    active_flows = {}
    completed_rows = []

    tcp443_packets_seen = 0
    first_seen_by_base_key = {}

    with open(pcap_path, "rb") as f:
        reader = open_pcap_reader(f, pcap_path)
        datalink = get_reader_datalink(reader)

        print(f"  Datalink usado na extração: {datalink}")
        print(f"  Modo de amostragem: {SAMPLE_MODE}")

        if SAMPLE_MODE == "flow":
            print(f"  Timeout de fluxo: {FLOW_TIMEOUT_SECONDS} segundos")

        if SAMPLE_MODE == "window":
            print(f"  Janela temporal: {WINDOW_SECONDS} segundos")

        for ts, buf in reader:
            try:
                ip = decode_ip_packet(buf, datalink)

                if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    continue

                tcp = ip.data

                if not isinstance(tcp, dpkt.tcp.TCP):
                    continue

                src_ip, dst_ip = get_ip_addresses(ip)

                if not src_ip or not dst_ip:
                    continue

                src_port = int(tcp.sport)
                dst_port = int(tcp.dport)

                if src_port == 443 or dst_port == 443:
                    tcp443_packets_seen += 1
                else:
                    continue

                packet_length = get_packet_length(buf, ip)

                if dst_port == 443:
                    base_key = (src_ip, dst_ip, src_port, 443)
                    direction = "src2dst"

                elif src_port == 443:
                    base_key = (dst_ip, src_ip, dst_port, 443)
                    direction = "dst2src"

                else:
                    continue

                current_ts = float(ts)

                if SAMPLE_MODE == "window":
                    if base_key not in first_seen_by_base_key:
                        first_seen_by_base_key[base_key] = current_ts

                    base_first_ts = first_seen_by_base_key[base_key]
                    window_id = int((current_ts - base_first_ts) // WINDOW_SECONDS)

                    sample_key = (base_key, window_id)

                    flow = active_flows.get(sample_key)

                    if flow is None:
                        flow = make_flow_record()
                        flow["base_key"] = base_key
                        active_flows[sample_key] = flow

                    append_packet_to_flow(flow, direction, packet_length, current_ts)
                    continue

                sample_key = base_key
                flow = active_flows.get(sample_key)

                if flow is None:
                    flow = make_flow_record()
                    flow["base_key"] = base_key
                    active_flows[sample_key] = flow

                else:
                    last_ts = flow["last_ts"]

                    if last_ts is not None:
                        gap = current_ts - float(last_ts)

                        if gap > FLOW_TIMEOUT_SECONDS:
                            completed_rows.append(finalize_flow_to_row(flow, tls_index))

                            flow = make_flow_record()
                            flow["base_key"] = base_key
                            active_flows[sample_key] = flow

                append_packet_to_flow(flow, direction, packet_length, current_ts)

            except Exception:
                continue

    for flow in active_flows.values():
        completed_rows.append(finalize_flow_to_row(flow, tls_index))

    print(f"  Pacotes TCP/443 vistos na extração: {tcp443_packets_seen}")
    print(f"  Amostras TCP/443 finalizadas: {len(completed_rows)}")

    if not completed_rows:
        return pd.DataFrame(columns=DOH_FEATURES)

    df = pd.DataFrame(completed_rows)

    for col in DOH_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    return df[DOH_FEATURES]


def fill_missing_dataset_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche valores ausentes sem remover colunas.
    """

    if not FILL_MISSING_VALUES:
        return df

    df = df.copy()

    tls_cols = ["sni", "alpn", "ja3"]
    numeric_cols = [c for c in DOH_FEATURES if c not in tls_cols]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in tls_cols:
        if col in df.columns:
            df[col] = df[col].fillna(UNKNOWN_TLS_VALUE).replace("", UNKNOWN_TLS_VALUE)

    return df


# ============================================================
# ESCRITA
# ============================================================

def append_dataset(csv_path: str, df: pd.DataFrame, header_written: bool) -> bool:
    mode = "a" if header_written else "w"

    df.to_csv(
        csv_path,
        index=False,
        mode=mode,
        header=not header_written,
        encoding="utf-8",
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():
    for fp in [OUT_DOH, DOH_REPORT, STATS_DOH]:
        if os.path.exists(fp):
            os.remove(fp)

    header_written = False
    report_rows = []

    arquivos = [
        f for f in os.listdir(pasta_pcap)
        if f.lower().endswith((".pcap", ".pcapng"))
    ]

    if not arquivos:
        print("Nenhum arquivo .pcap ou .pcapng encontrado.")
        return

    print(f"Total de arquivos PCAP/PCAPNG encontrados: {len(arquivos)}")
    print(f"Pasta de entrada: {pasta_pcap}")
    print(f"Pasta de saída:   {pasta_saida}")
    print(f"Timeout de fluxo: {FLOW_TIMEOUT_SECONDS} segundos")
    print(f"Modo de amostragem: {SAMPLE_MODE}")

    if SAMPLE_MODE == "window":
        print(f"Janela temporal: {WINDOW_SECONDS} segundos")

    print("")

    for nome_arquivo in arquivos:
        if not is_doh_file(nome_arquivo):
            print(f"[SKIP] {nome_arquivo} não parece ser DoH pelo nome.")
            continue

        caminho_pcap = os.path.join(pasta_pcap, nome_arquivo)
        label = detect_label(nome_arquivo)

        print(f"[{datetime.now()}] Processando DoH: {nome_arquivo} | label={label}")

        try:
            tls_index, clienthello_packets_seen, tcp443_prescan_seen = build_tls_index_for_doh(
                caminho_pcap
            )

            df = extract_doh_features_from_pcap(caminho_pcap, tls_index)

            if df.empty:
                print(f"  ⚠️ Nenhuma amostra TCP/443 extraída de {nome_arquivo}")
                continue

            df = fill_missing_dataset_values(df)

            df["label"] = label
            df = df[FINAL_COLUMNS]

            tls_missing_mask = (
                df[["sni", "alpn", "ja3"]].isna()
                | df[["sni", "alpn", "ja3"]].eq(UNKNOWN_TLS_VALUE)
            ).any(axis=1)

            missing_any_tls = int(tls_missing_mask.sum())
            tls_ok = len(df) - int(missing_any_tls)

            header_written = append_dataset(
                OUT_DOH,
                df,
                header_written,
            )

            report_rows.append({
                "pcap": nome_arquivo,
                "label": label,
                "samples_extracted": int(len(df)),
                "tcp443_packets_seen_prescan": int(tcp443_prescan_seen),
                "clienthello_packets_seen_heuristic": int(clienthello_packets_seen),
                "tls_index_entries": int(len(tls_index)),
                "samples_missing_any_tls_field": int(missing_any_tls),
                "samples_tls_ok": int(tls_ok),
                "sample_mode": SAMPLE_MODE,
                "flow_timeout_seconds": float(FLOW_TIMEOUT_SECONDS),
                "window_seconds": float(WINDOW_SECONDS) if SAMPLE_MODE == "window" else np.nan,
            })

            print(f"  [OK] {len(df)} amostras adicionadas ao dataset_DoH.csv")
            print(f"       TCP/443 vistos no pré-scan: {tcp443_prescan_seen}")
            print(f"       ClientHello vistos: {clienthello_packets_seen}")
            print(f"       TLS indexados: {len(tls_index)}")
            print(f"       TLS OK: {tls_ok}")
            print(f"       TLS com algum campo ausente: {missing_any_tls}")

        except Exception as e:
            print(f"  [ERRO] {nome_arquivo}: {e}")

    if report_rows:
        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(DOH_REPORT, index=False, encoding="utf-8")

        stats_df = report_df.groupby("label").agg(
            sample_count=("samples_extracted", "sum"),
            tcp443_packets_seen_prescan=("tcp443_packets_seen_prescan", "sum"),
            clienthello_packets_seen_heuristic=("clienthello_packets_seen_heuristic", "sum"),
            tls_index_entries=("tls_index_entries", "sum"),
            samples_missing_any_tls_field=("samples_missing_any_tls_field", "sum"),
            samples_tls_ok=("samples_tls_ok", "sum"),
        ).reset_index()

        stats_df.to_csv(STATS_DOH, index=False, encoding="utf-8")

    print("")
    print("Finalizado.")
    print(f"Dataset DoH: {OUT_DOH}")
    print(f"Report DoH:  {DOH_REPORT}")
    print(f"Stats DoH:   {STATS_DOH}")

    if os.path.exists(OUT_DOH):
        check = pd.read_csv(OUT_DOH, nrows=5)

        print("")
        print("Colunas geradas no dataset_DoH.csv:")
        print(check.columns.tolist())

        print("")
        print("Amostra:")
        print(check.head())

        print("")
        print("Distribuição inicial de labels:")
        full_labels = pd.read_csv(OUT_DOH, usecols=["label"])
        print(full_labels["label"].value_counts())

        print("")
        print("JA3 únicos:")
        ja3_check = pd.read_csv(OUT_DOH, usecols=["ja3"])
        print(ja3_check["ja3"].nunique(dropna=True))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extrator de features Do53 a partir de PCAP/PCAPNG.

Pipeline:
1) Lê arquivos PCAP/PCAPNG.
2) Filtra pacotes DNS convencionais na porta 53.
3) Extrai atributos léxicos, estruturais e DNS.
4) Calcula estatísticas agregadas por domínio-base.
5) Gera dataset_Do53.csv com 29 features + label.

Bibliotecas principais:
- dpkt: leitura e parsing dos pacotes.
- NumPy: cálculo estatístico.
- pandas: organização tabular e exportação CSV.
"""

import os
import re
import math
import socket
import unicodedata
from collections import defaultdict

import dpkt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

pasta_pcap = r"C:\Users\cborges\OneDrive - Fortinet\Documents\Studies\Doctorade UFU\Research\Development\Dataset\My Dataset\PCAPs\Do53"

pasta_saida = r"C:\Users\cborges\OneDrive - Fortinet\Documents\Studies\Doctorade UFU\Research\Development\Dataset\My Dataset\CSVs\Do53\Versao_Final"

os.makedirs(pasta_saida, exist_ok=True)

OUT_DO53 = os.path.join(pasta_saida, "dataset_Do53.csv")
STATS_DO53 = os.path.join(pasta_saida, "stats_Do53.csv")

DO53_FEATURES = [
    "query_length",
    "subdomain_length",
    "num_labels",
    "max_label_length",
    "avg_label_length",
    "entropy",
    "ratio_digits",
    "ratio_uppercase",
    "ratio_lowercase",
    "ratio_special_chars",
    "rr_type",
    "query_class",
    "ttl",
    "contains_base64_pattern",
    "contains_hex_pattern",
    "contains_dictionary_word",
    "ratio_vowels",
    "ratio_consonants",
    "num_distinct_characters",
    "mean_query_length_per_domain",
    "std_query_length_per_domain",
    "mean_entropy_per_domain",
    "num_unique_ips",
    "num_unique_asns",
    "num_unique_ttls",
    "mean_ttl",
    "var_ttl",
    "num_queries_per_domain",
    "ratio_failed_queries",
]

FINAL_COLUMNS = DO53_FEATURES + ["label"]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalize_filename(nome_arquivo: str) -> str:
    nome = nome_arquivo.lower()
    nome = unicodedata.normalize("NFKD", nome)
    nome = nome.encode("ascii", "ignore").decode("ascii")
    return nome


def detect_label(nome_arquivo: str) -> str:
    """
    Define o label com base no nome do arquivo.
    Ajuste os tokens conforme sua nomenclatura real.
    """
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


def is_do53_file(nome_arquivo: str) -> bool:
    nome = normalize_filename(nome_arquivo)

    if "do53" in nome:
        return True

    if "dns" in nome:
        return True

    if "_53" in nome or "-53" in nome:
        return True

    return True


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


def clean_dns_name(name) -> str:
    if isinstance(name, bytes):
        name = name.decode(errors="ignore")

    name = str(name).strip().lower()

    if name.endswith("."):
        name = name[:-1]

    return name


def get_labels(domain: str):
    domain = clean_dns_name(domain)

    if not domain:
        return []

    return [x for x in domain.split(".") if x]


def get_base_domain(domain: str) -> str:
    """
    Estratégia simples: usa os dois últimos labels.
    Exemplo:
    a.b.example.com -> example.com

    Para maior precisão em domínios como .com.br, pode-se integrar
    uma Public Suffix List, mas isso exigiria outra dependência.
    """
    labels = get_labels(domain)

    if len(labels) >= 2:
        return ".".join(labels[-2:])

    if labels:
        return labels[0]

    return "unknown"


def calculate_entropy(text: str) -> float:
    text = text.replace(".", "")

    if not text:
        return 0.0

    freq = defaultdict(int)

    for char in text:
        freq[char] += 1

    entropy = 0.0
    length = len(text)

    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)

    return float(entropy)


def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else 0.0


def safe_std(values):
    return float(np.std(values, ddof=0)) if len(values) > 0 else 0.0


def safe_var(values):
    return float(np.var(values, ddof=0)) if len(values) > 0 else 0.0


def ratio_digits(text: str) -> float:
    text = text.replace(".", "")

    if not text:
        return 0.0

    return sum(c.isdigit() for c in text) / len(text)


def ratio_uppercase(text: str) -> float:
    raw = text.replace(".", "")

    if not raw:
        return 0.0

    return sum(c.isupper() for c in raw) / len(raw)


def ratio_lowercase(text: str) -> float:
    raw = text.replace(".", "")

    if not raw:
        return 0.0

    return sum(c.islower() for c in raw) / len(raw)


def ratio_special_chars(text: str) -> float:
    raw = text.replace(".", "").replace("-", "")

    if not raw:
        return 0.0

    return sum(not c.isalnum() for c in raw) / len(raw)


def ratio_vowels(text: str) -> float:
    raw = re.sub(r"[^a-zA-Z]", "", text)

    if not raw:
        return 0.0

    vowels = set("aeiouAEIOU")

    return sum(c in vowels for c in raw) / len(raw)


def ratio_consonants(text: str) -> float:
    raw = re.sub(r"[^a-zA-Z]", "", text)

    if not raw:
        return 0.0

    vowels = set("aeiouAEIOU")

    return sum(c not in vowels for c in raw) / len(raw)


def contains_base64_pattern(text: str) -> int:
    """
    Heurística simples para identificar sequências compatíveis com Base32/Base64.
    """
    raw = text.replace(".", "").replace("-", "")

    if len(raw) < 20:
        return 0

    pattern = r"[A-Za-z0-9+/=]{20,}"

    return int(bool(re.search(pattern, raw)))


def contains_hex_pattern(text: str) -> int:
    """
    Heurística para sequências hexadecimais longas.
    """
    raw = text.replace(".", "").replace("-", "")

    pattern = r"[a-fA-F0-9]{16,}"

    return int(bool(re.search(pattern, raw)))


def contains_dictionary_word(text: str) -> int:
    """
    Heurística simples para verificar presença de palavras comuns.
    Ajuste a lista conforme o idioma/contexto do tráfego benigno.
    """
    words = {
        "google",
        "cloud",
        "login",
        "mail",
        "office",
        "microsoft",
        "windows",
        "apple",
        "facebook",
        "amazon",
        "cdn",
        "api",
        "www",
        "portal",
        "update",
        "download",
        "security",
        "fortinet",
    }

    labels = get_labels(text)

    for label in labels:
        for word in words:
            if word in label:
                return 1

    return 0


def extract_answer_ips_and_ttls(dns):
    ips = []
    ttls = []

    sections = []

    try:
        sections.extend(dns.an)
    except Exception:
        pass

    try:
        sections.extend(dns.ar)
    except Exception:
        pass

    for rr in sections:
        try:
            if hasattr(rr, "ttl"):
                ttls.append(float(rr.ttl))
        except Exception:
            pass

        try:
            if rr.type == dpkt.dns.DNS_A:
                ips.append(socket.inet_ntoa(rr.rdata))
            elif rr.type == dpkt.dns.DNS_AAAA:
                ips.append(socket.inet_ntop(socket.AF_INET6, rr.rdata))
        except Exception:
            continue

    return ips, ttls


def parse_dns_from_udp_or_tcp(transport, src_port: int, dst_port: int):
    """
    Retorna objeto DNS se for pacote DNS válido.
    Suporta UDP/53 e TCP/53.

    Em DNS sobre TCP, os dois primeiros bytes indicam o tamanho da mensagem DNS.
    """
    try:
        if isinstance(transport, dpkt.udp.UDP):
            if src_port != 53 and dst_port != 53:
                return None

            if not transport.data:
                return None

            return dpkt.dns.DNS(transport.data)

        if isinstance(transport, dpkt.tcp.TCP):
            if src_port != 53 and dst_port != 53:
                return None

            if not transport.data or len(transport.data) < 3:
                return None

            # DNS over TCP possui prefixo de 2 bytes com tamanho.
            dns_payload = transport.data[2:]

            return dpkt.dns.DNS(dns_payload)

    except Exception:
        return None

    return None


# ============================================================
# EXTRAÇÃO DO53
# ============================================================

def extract_do53_records_from_pcap(pcap_path: str, label: str) -> pd.DataFrame:
    rows = []

    dns_packets_seen = 0
    dns_queries_seen = 0

    with open(pcap_path, "rb") as f:
        reader = open_pcap_reader(f, pcap_path)
        datalink = get_reader_datalink(reader)

        print(f"  Datalink usado na extração: {datalink}")

        for ts, buf in reader:
            try:
                ip = decode_ip_packet(buf, datalink)

                if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    continue

                transport = ip.data

                if not isinstance(transport, (dpkt.udp.UDP, dpkt.tcp.TCP)):
                    continue

                src_port = int(transport.sport)
                dst_port = int(transport.dport)

                if src_port != 53 and dst_port != 53:
                    continue

                dns = parse_dns_from_udp_or_tcp(transport, src_port, dst_port)

                if dns is None:
                    continue

                dns_packets_seen += 1

                if not hasattr(dns, "qd") or len(dns.qd) == 0:
                    continue

                answer_ips, answer_ttls = extract_answer_ips_and_ttls(dns)

                if answer_ttls:
                    ttl_value = float(answer_ttls[0])
                else:
                    ttl_value = 0.0

                rcode = int(getattr(dns, "rcode", 0))
                is_failed = 1 if rcode != 0 else 0

                for question in dns.qd:
                    try:
                        qname = clean_dns_name(question.name)

                        if not qname:
                            continue

                        dns_queries_seen += 1

                        labels = get_labels(qname)
                        base_domain = get_base_domain(qname)

                        query_length = len(qname)
                        subdomain_length = len(labels[0]) if labels else 0
                        num_labels = len(labels)
                        label_lengths = [len(x) for x in labels]

                        max_label_length = max(label_lengths) if label_lengths else 0
                        avg_label_length = safe_mean(label_lengths)

                        entropy_value = calculate_entropy(qname)

                        row = {
                            "query": qname,
                            "base_domain": base_domain,
                            "timestamp": float(ts),

                            "query_length": float(query_length),
                            "subdomain_length": float(subdomain_length),
                            "num_labels": float(num_labels),
                            "max_label_length": float(max_label_length),
                            "avg_label_length": float(avg_label_length),
                            "entropy": float(entropy_value),
                            "ratio_digits": float(ratio_digits(qname)),
                            "ratio_uppercase": float(ratio_uppercase(question.name)),
                            "ratio_lowercase": float(ratio_lowercase(question.name)),
                            "ratio_special_chars": float(ratio_special_chars(qname)),
                            "rr_type": float(getattr(question, "type", 0)),
                            "query_class": float(getattr(question, "cls", 0)),
                            "ttl": float(ttl_value),
                            "contains_base64_pattern": float(contains_base64_pattern(qname)),
                            "contains_hex_pattern": float(contains_hex_pattern(qname)),
                            "contains_dictionary_word": float(contains_dictionary_word(qname)),
                            "ratio_vowels": float(ratio_vowels(qname)),
                            "ratio_consonants": float(ratio_consonants(qname)),
                            "num_distinct_characters": float(len(set(qname.replace(".", "")))),

                            # Campos auxiliares para agregação por domínio.
                            "answer_ips": answer_ips,
                            "answer_ttls": answer_ttls,
                            "is_failed": int(is_failed),

                            "label": label,
                        }

                        rows.append(row)

                    except Exception:
                        continue

            except Exception:
                continue

    print(f"  Pacotes DNS vistos: {dns_packets_seen}")
    print(f"  Consultas DNS extraídas: {dns_queries_seen}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def add_domain_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula as features agregadas por domínio-base:
    - mean_query_length_per_domain
    - std_query_length_per_domain
    - mean_entropy_per_domain
    - num_unique_ips
    - num_unique_asns
    - num_unique_ttls
    - mean_ttl
    - var_ttl
    - num_queries_per_domain
    - ratio_failed_queries
    """

    if df.empty:
        return df

    aggregate_rows = []

    for base_domain, group in df.groupby("base_domain"):
        query_lengths = group["query_length"].astype(float).tolist()
        entropies = group["entropy"].astype(float).tolist()
        ttls = group["ttl"].astype(float).tolist()
        failed = group["is_failed"].astype(int).tolist()

        unique_ips = set()
        unique_ttls = set()

        for _, row in group.iterrows():
            for ip_addr in row.get("answer_ips", []):
                unique_ips.add(ip_addr)

            for ttl in row.get("answer_ttls", []):
                unique_ttls.add(float(ttl))

        # Sem consulta externa de ASN para manter o pipeline offline/reprodutível.
        # Caso você tenha um CSV de cache IP->ASN, dá para integrar aqui.
        unique_asns = set()

        aggregate_rows.append({
            "base_domain": base_domain,
            "mean_query_length_per_domain": safe_mean(query_lengths),
            "std_query_length_per_domain": safe_std(query_lengths),
            "mean_entropy_per_domain": safe_mean(entropies),
            "num_unique_ips": float(len(unique_ips)),
            "num_unique_asns": float(len(unique_asns)),
            "num_unique_ttls": float(len(unique_ttls)),
            "mean_ttl": safe_mean(ttls),
            "var_ttl": safe_var(ttls),
            "num_queries_per_domain": float(len(group)),
            "ratio_failed_queries": safe_mean(failed),
        })

    df_agg = pd.DataFrame(aggregate_rows)

    df = df.merge(df_agg, on="base_domain", how="left")

    return df


def finalize_do53_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mantém somente as 29 features finais + label.
    Preenche ausentes com 0.0.
    """

    if df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    for col in DO53_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    for col in DO53_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["label"] = df["label"].astype(str)

    return df[FINAL_COLUMNS]


def create_stats(df: pd.DataFrame, output_path: str):
    if df.empty:
        print("Nenhuma amostra para gerar estatísticas.")
        return

    counts = df["label"].value_counts().rename("count")
    percentages = df["label"].value_counts(normalize=True).mul(100).rename("percentage")

    stats = pd.concat([counts, percentages], axis=1)
    stats.to_csv(output_path, encoding="utf-8")

    print("\nDistribuição das classes:")
    print(stats)


# ============================================================
# MAIN
# ============================================================

def main():
    if os.path.exists(OUT_DO53):
        os.remove(OUT_DO53)

    if os.path.exists(STATS_DO53):
        os.remove(STATS_DO53)

    all_dfs = []

    arquivos = [
        f for f in os.listdir(pasta_pcap)
        if f.lower().endswith((".pcap", ".pcapng")) and is_do53_file(f)
    ]

    arquivos.sort()

    print(f"Arquivos PCAP/PCAPNG encontrados: {len(arquivos)}")

    for arquivo in arquivos:
        pcap_path = os.path.join(pasta_pcap, arquivo)
        label = detect_label(arquivo)

        print("\n============================================================")
        print(f"Processando arquivo: {arquivo}")
        print(f"Label detectado: {label}")
        print("============================================================")

        df_file = extract_do53_records_from_pcap(pcap_path, label)

        if df_file.empty:
            print("  Nenhuma consulta DNS extraída deste arquivo.")
            continue

        all_dfs.append(df_file)

    if not all_dfs:
        print("Nenhum dado Do53 foi extraído.")
        return

    print("\nConcatenando dados extraídos...")
    df = pd.concat(all_dfs, ignore_index=True)

    print(f"Linhas antes das agregações: {len(df)}")

    print("Calculando agregações por domínio-base...")
    df = add_domain_aggregates(df)

    print("Finalizando dataset...")
    df_final = finalize_do53_dataframe(df)

    print(f"Shape final: {df_final.shape}")

    df_final.to_csv(OUT_DO53, index=False, encoding="utf-8")

    print(f"\nDataset salvo em:")
    print(OUT_DO53)

    create_stats(df_final, STATS_DO53)

    print(f"\nEstatísticas salvas em:")
    print(STATS_DO53)

    print("\nFeatures finais:")
    for i, col in enumerate(DO53_FEATURES, start=1):
        print(f"{i:02d}. {col}")

    print("\nProcessamento concluído.")


if __name__ == "__main__":
    main()
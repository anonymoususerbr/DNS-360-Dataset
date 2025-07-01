# DNS-360-Dataset

This repository contains a curated dataset of DNS exfiltration traffic, including DNS over UDP (Do53) and DNS over HTTPS (DoH) examples. The dataset is intended for use in academic research, threat detection benchmarking, and machine learning experimentation.

#Contents

- `data/` — Labeled DNS query/response logs (Do53 and DoH)
- `metadata/` — Descriptions, label definitions, collection methodology
- `scripts/` — Example parsing and analysis tools

#Dataset Description

The dataset includes:

- Normal and exfiltration DNS traffic samples
- Features extracted from packet-level and session-level data
- Separate folders for Do53 and DoH traffic
- Metadata with timestamps, domains, query types, and label annotations


#Use Cases

- Anomaly detection using statistical or ML models
- Benchmarking DNS-based intrusion detection systems
- Comparative analysis of encrypted vs unencrypted exfiltration methods

#License

This dataset is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

You are free to:

- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.

For full details, see the [LICENSE](./LICENSE) file or visit:  
https://creativecommons.org/licenses/by-nc/4.0/

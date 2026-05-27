# DNS-360-Dataset

This repository contains a curated dataset and experimental pipeline for studying DNS-based data exfiltration in two scenarios:

- **Do53**: conventional DNS traffic over port 53.
- **DoH**: DNS over HTTPS traffic over TCP/443.

The repository is intended for academic research, threat detection benchmarking, feature engineering, and machine learning experiments focused on DNS-based data exfiltration.

## Repository Structure

```text
DNS-360-Dataset/
│
├── data/
│   ├── pcaps/
│   │   ├── Do53/
│   │   └── DoH/
│   │
│   └── csv/
│       ├── Do53/
│       └── DoH/
│
├── dns-exfiltration-ml-pipeline/
│   ├── pcap_to_flow_do53.py
│   ├── pcap_to_flow_doh.py
│   ├── ML_Do53_RF_V3.py
│   ├── ML_Do53_SVM_V1.py
│   ├── ML_Do53_XGBoost_V1.py
│   ├── ML_DoH_RF_V1.py
│   ├── ML_DoH_SVM_V1.py
│   └── ML_DoH_XGBoost_V1.py
│
├── LICENSE
└── README.md

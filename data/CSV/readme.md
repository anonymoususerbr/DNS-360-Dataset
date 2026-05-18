## Dataset Overview

This repository provides two complementary datasets for research on DNS-based data exfiltration detection: one focused on conventional DNS traffic over port 53 and another focused on DNS over HTTPS (DoH) traffic over TCP/443.

Both datasets were generated under a controlled experimental methodology, including benign and exfiltration scenarios, PCAP traffic capture, reproducible PCAP-to-Flow processing, explicit labeling, and protocol-specific feature extraction.

The Do53 dataset supports the analysis of exfiltration patterns in clear-text DNS traffic, where query structure and DNS metadata are directly observable. The DoH dataset supports the analysis of exfiltration in encrypted DNS traffic, where detection relies mainly on flow-level, directional, size, volume, and timing characteristics.

Together, these datasets provide a consistent basis for evaluating machine learning approaches for data exfiltration detection across both traditional and encrypted DNS environments.

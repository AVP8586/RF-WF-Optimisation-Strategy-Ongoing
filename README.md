<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">RF-WF-OPTIMISATION-STRATEGY-ONGOING</h1></p>
<p align="center">
	<em><code>❯ REPLACE-ME</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/AVP8586/RF-WF-Optimisation-Strategy-Ongoing?style=default&logo=opensourceinitiative&logoColor=white&color=36ff00" alt="license">
	<img src="https://img.shields.io/github/last-commit/AVP8586/RF-WF-Optimisation-Strategy-Ongoing?style=default&logo=git&logoColor=white&color=36ff00" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/AVP8586/RF-WF-Optimisation-Strategy-Ongoing?style=default&color=36ff00" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/AVP8586/RF-WF-Optimisation-Strategy-Ongoing?style=default&color=36ff00" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
- [ Project Roadmap](#-project-roadmap)

---

##  Overview

RF-WF-Optimisation-Strategy-Ongoing is an advanced trading algorithm that combines Reinforcement Learning (RL) and Walk Forward Optimisation (WFO) approaches. The strategy is designed to automate and enhance trading decisions by adapting to dynamic market conditions over time, with a particular focus on cryptocurrency markets such as Bitcoin (BTC).

---

##  Features

1. Integrated RL & WFO: Learns from historical and ongoing market data, adapting policy as new data arrives.
2. Modular Implementation: Separate modules for RL agent (PPO.py), trading logic (Trading.py), parameter management, and data handling.
3. Custom Dataset Support: Ready to use with cryptocurrency price data (e.g., BTC_2019_2023_6h.csv).
4. Flexible Configuration: Easy-to-modify parameters for experimentation and tuning.
5. Scalable and Extensible: Foundation for adding new RL algorithms, assets, or timeframes.

---

##  Project Structure

```sh
└── RF-WF-Optimisation-Strategy-Ongoing/
    ├── PPO.py
    ├── Trading.py
    ├── data
    │   └── BTC_2019_2023_6h.csv
    └── parameters.py
```


---
##  Project Roadmap

- [ ] Extend asset and timeframe support
- [ ] Integrate additional RL algorithms
- [ ] Enhance backtesting and walk-forward analysis features
- [ ] Deployment and live trading integrations

---

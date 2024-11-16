[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmaxerh%2FTSAD_dataset_inspection&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Time Series Anomaly Detection datasets inspection

## Datasets
- Server Machine Dataset (SMD): https://github.com/NetManAIOps/OmniAnomaly
- Pooled Server Metrics (PSM): https://github.com/eBay/RANSynCoders
- Secure Water Treatment (SWaT): https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
- Water Distribution (WADI): https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
- Soil Moisture Active Passive (SMAP)
- Mars Science Laboratory (MSL)
```shell
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip

cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

## How To

```shell
python main.py --lags 1024 --datasets SMD_PSM
```
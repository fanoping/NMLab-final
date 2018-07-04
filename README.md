# NMLab Final Project
Networking and Multimedia Lab (2018 Spring) at National Taiwan University

Reference: [[Packettotal](https://packettotal.com/)]
## Flow Detection
convert network flow to csv files
* Usage
    * Real-time
    ```
    python pcap2csv.py realtime --packet_cnt[int]
    ```
    * Off-time
    ```
     python pcap2csv.py offtime --input[pcap directory] --output[csv directory]
    ```
## Flow Classification
* **Usage**: 

    choose arch from [...] (K-nearest neighboring, decision tree, random forest for Scenario B)
    
    modify the config.json if other attributes specified
 
    * Scenario A
        
        ```
            python classifierA.py -k 10 --input-csv [csv_file] --config [config.json] --arch [knn, tree] 
        ```
    * Scenario B
    
        ```
            python classifierB.py -k 10 --input-csv [csv_file] --config [config.json] --arch [knn, tree, forest] 
        ```

## 分工表
    
   * 温明浩: 前端網頁&後端server建立、後期debug

   * 陳柏文: Python程式修飾、程式與server串接、後期debug

   * 徐彥旻: 報告製作、Python程式修飾

   * 周晁德: model撰寫&訓練

   * 蔡昕宇: model撰寫&訓練

   * 黃平瑋: pcap檔案分析與轉換

   * 劉記良: 報告製作

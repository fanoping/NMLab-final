# NMLab Final Project
Networking and Multimedia Lab (2018 Spring) at National Taiwan University

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

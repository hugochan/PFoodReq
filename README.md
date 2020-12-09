# PFoodReq

Code & data accompanying the WSDM 2021 paper "Personalized Food Recommendation as Constrained Question Answering over a Large-scale Food Knowledge Graph".


## Architecture

![PFoodReq architecture.](images/arch.png)




## Prerequisites
This code is written in python 3. You will need to install a few python packages in order to run the code.
We recommend you to use `virtualenv` to manage your python packages and environments.
Please take the following steps to create a python virtual environment.

* If you have not installed `virtualenv`, install it with ```pip install virtualenv```.
* Create a virtual environment with ```virtualenv venv```.
* Activate the virtual environment with `source venv/bin/activate`.
* Install the package requirements with `pip install -r requirements.txt`.


You will need to download the partial FoodKG used in our experiments from [here](https://1drv.ms/u/s!AjiSpuwVTt09gVEK2dFDBitRvQ0l?e=tGHXuv) and move the `recipe_kg` folder to the `data` folder in this repo.


## Create a KBQA dataset

* Go to `data_builder/src` folder, run the following cmd:

	```
	python generate_all_qa.py -recipe ../../data/recipe_kg/recipe_kg.json -o ../../data/kbqa_data/ -out_of_domain_ratio 0.1 -split_ratio 0.6 0.2
	```

You can also download our generated benchmark from [here](https://1drv.ms/u/s!AjiSpuwVTt09gVEK2dFDBitRvQ0l?e=tGHXuv) and move the `kbqa_data` folder to the `data` folder in this repo. In the downloaded `kbqa_data` folder, you will also see additional files used in the KBQA+RecipeSim setting.

## Run a KBQA system

### Preprocess the data

* Download the pretrained Glove word ebeddings [glove.840B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and move it to the `data` folder in this repo.

* Go to the `BAMnet/src` folder

* Preprocess data
	* Vectorize data
	```
	python build_all_data.py -data_dir ../../data/kbqa_data/ -kb_path ../../data/recipe_kg/recipe_kg.json -out_dir ../data/kbqa
	```
  In the message printed out, your will see some data statistics such as `vocab_size`, `num_ent_types`, and `num_relations`. These numbers will be used later when modifying the config file.
	* Fetch the pretrained Glove vectors for our vocabulary.

      ```
      python build_pretrained_w2v.py -emb ../../data/glove.840B.300d.txt -data_dir ../data/kbqa/ -out ../data/kbqa/glove_pretrained_300d_w2v.npy
      ```




* Preprocess data w/o kg augmentation
	* Vectorize data
	```
    python build_all_data.py -data_dir ../../data/kbqa_data/ -kb_path ../../data/recipe_kg/recipe_kg.json -out_dir ../data/kbqa_no_ka  --no_kg_augmentation
    ```
	* Fetch the pretrained Glove vectors for our vocabulary.
	```
   	 python build_pretrained_w2v.py -emb ../../data/glove.840B.300d.txt -data_dir ../data/kbqa_no_ka/ -out ../data/kbqa_no_ka/glove_pretrained_300d_w2v.npy
	```



* Preprocess data w/o query expansion
	* Vectorize data
    ```
    python build_all_data.py -data_dir ../../data/kbqa_data/ -kb_path ../../data/recipe_kg/recipe_kg.json -out_dir ../data/kbqa_no_qe --no_query_expansion
    ```
	* Fetch the pretrained Glove vectors for our vocabulary.
    ```
    python build_pretrained_w2v.py -emb ../../data/glove.840B.300d.txt -data_dir ../data/kbqa_no_qe/ -out ../data/kbqa_no_qe/glove_pretrained_300d_w2v.npy
    ```



* Preprocess data w/o query expansion & kg augmentation
	* Vectorize data
	```
	python build_all_data.py -data_dir ../../data/kbqa_data/ -kb_path ../../data/recipe_kg/recipe_kg.json -out_dir ../data/kbqa_no_qe_ka --no_query_expansion --no_kg_augmentation
	```
	* Fetch the pretrained Glove vectors for our vocabulary.
    ```
    python build_pretrained_w2v.py -emb ../../data/glove.840B.300d.txt -data_dir ../data/kbqa_no_qe_ka/ -out ../data/kbqa_no_qe_ka/glove_pretrained_300d_w2v.npy
	```



### Tran/test a KBQA system

* Modify the config file `BAMnet/src/config/kbqa.yml` to suit your needs. Note that you can start with modifying only the data folder and vocab size (e.g., `data_dir`, `kb_path`,
`pre_word2vec`, `vocab_size`, `num_ent_types`, and `num_relations`), and leave other variables as they are.



* Train the KBQA model.

	```
	python train.py -config config/pfoodreq.yml
	```

* Test the KBQA model:

	```
	python run_online.py -config config/pfoodreq.yml
	```



* Train the KBQA+RecipeSim model.

    ```
    python train.py -config config/pfoodreq_similar_recipes.yml
    ```

* Test the KBQA+RecipeSim model:

    ```
    python run_online.py -config config/pfoodreq_similar_recipes.yml
    ```


## Reference

If you found this code useful, please consider citing the following paper:

Yu Chen, Ananya Subburathinam, Ching-Hua Chen and Mohammed J. Zaki. **"Personalized Food Recommendation as Constrained Question Answering over a Large-scale Food Knowledge Graph."** In *Proceedings of the 14th International Conference on Web Search and Data Mining (WSDM 2021), Mar. 8-12, 2021.*


    @inproceedings{chen2021personalized,
    author    = {Chen, Yu and Subburathinam, Ananya and Chen, Ching-Hua and Zaki, Mohammed J.},
    title     = {Personalized Food Recommendation as Constrained Question Answering over a Large-scale Food Knowledge Graph},
    booktitle = {Proceedings of the 14th International Conference on Web Search and Data Mining},
    month = {Mar. 8-12,},
    year      = {2021}}






Note that we use the BAMnet model as our KBQA system in this application. For more details about the BAMnet model, please refer to the [original paper](https://arxiv.org/abs/1903.02188). For more details about the FoodKG, please refer to the [original paper](https://link.springer.com/chapter/10.1007/978-3-030-30796-7_10).


from exptfm.inference import EmbeddingModel, StaticEmbeddingModel
from exptfm.similarity import compute_cosine_similarity, compute_euclidean_distance, compute_icmb
from exptfm.correlation import compute_pearson_correlation, compute_spearman_correlation
from exptfm.datasets import load_dataset_as_pandas
import pandas as pd
import torch
import os
from sklearn.preprocessing import MinMaxScaler 
import numpy as np

class Experiment():
    def __init__(self):
        self.expdir = "exptfm"
        self.results = "results"
        self.model_name = ""
        self.number_of_layers = 0
        print("Loading dataset...")
        self.ds = load_dataset_as_pandas(dataset="stsb", subset="validation")
        self.embeddings = []
    
    def set_model(self, model_name, number_of_layers):
        self.model_name = model_name.replace("/", "-")
        self.number_of_layers = number_of_layers        
        print("Loading embedding model '"+self.model_name+"'...")
        self.model = EmbeddingModel(model_name)
        self.embeddings = []

    def get_file_name_for_model(self, suffix, ext):
        return self.model_name + "_" + suffix + ext

    def save_to_csv(self, target_df, suffix):
        file_name = self.get_file_name_for_model(suffix=suffix, ext=".csv")
        file_path = os.path.join(self.expdir, self.results, file_name)
        print("Saving file '"+file_path+"' to disk")
        target_df.to_csv(file_path, index=False)

    def save_to_disk(self, data, suffix):
        file_name = self.get_file_name_for_model(suffix=suffix, ext=".pt")
        file_path = os.path.join(self.expdir, self.results, file_name)
        print("Saving file '"+file_path+"' to disk")
        torch.save( data, file_path )

    # def compute_embeddings(self, num_layers, composition_functions):
    #     results = []
    #     embeddings_data = []
        
    #     for idx, row in self.ds.iterrows():
    #         s1, s2 = row["sentence1"], row["sentence2"]
    #         base_embedding_1 = self.model.get_embeddings([s1])
    #         base_embedding_2 = self.model.get_embeddings([s2])

    #         row_result = {
    #             "idx": idx,
    #             "sentence1": s1,
    #             "sentence2": s2,
    #             "label": row["label"],
    #             "label_ok": row["label_ok"],
    #             "model": self.model_name,
    #         }

    #         print("processing:", idx)

    #         for compf in composition_functions:
    #             for layer_n in range(num_layers):
    #                 e1 = self.model.get_embedding_from_layer_with_composition_function(base_embedding_1, compf, layer_n)
    #                 e2 = self.model.get_embedding_from_layer_with_composition_function(base_embedding_2, compf, layer_n)

    #                 # Calculate similarities
    #                 row_result[f"Cos{compf.capitalize()}{layer_n}"] = compute_cosine_similarity(e1, e2)
    #                 row_result[f"Euc{compf.capitalize()}{layer_n}"] = compute_euclidean_distance(e1, e2)
    #                 row_result[f"Icm{compf.capitalize()}{layer_n}"] = compute_icmb(e1, e2)

    #                 embeddings_data.append({
    #                     "idx": idx, 
    #                     "model": self.model_name, 
    #                     "compf": compf, 
    #                     "layer": layer_n,
    #                     "sentence1": e1, 
    #                     "sentence2": e2
    #                 })

    #         results.append(row_result)

    #     return results, embeddings_data

    # def normalize_and_align_similarity(self, results, num_layers, composition_functions):
    #     # Normalize and align metrics to similarity
    #     scaler = MinMaxScaler()
    #     normalized_results = []
    #     for row_result in results:
    #         normalized_row = row_result.copy()
    #         for compf in composition_functions:
    #             for layer_n in range(num_layers):
    #                 # Extract raw values
    #                 euc_key = f"Euc{compf.capitalize()}{layer_n}"
    #                 icm_key = f"Icm{compf.capitalize()}{layer_n}"

    #                 # Normalize Euclidean distance and ICM
    #                 values_to_normalize = np.array([
    #                     row_result.get(euc_key, 0), 
    #                     row_result.get(icm_key, 0)
    #                 ]).reshape(-1, 1)

    #                 normalized_values = scaler.fit_transform(values_to_normalize).flatten()
    #                 normalized_row[euc_key] = normalized_values[0]
    #                 normalized_row[icm_key] = normalized_values[1]

    #                 # Invert the normalized Euclidean distance to align it to similarity
    #                 normalized_row[euc_key] = 1 - normalized_row[euc_key]

    #         normalized_results.append(normalized_row)    

    #     return normalized_results    

    # def compute_correlations(self, num_layers, composition_functions, similarity_metrics):
    #     corrp = []
    #     for simf in similarity_metrics:
    #         for compf in composition_functions:
    #             for i in range(num_layers):
    #                 key = f"{simf}{compf.capitalize()}{i}"
    #                 corrp.append({
    #                     "model": self.model_name,
    #                     "type": "Transformer",
    #                     "composition": simf,
    #                     "similarity": compf.capitalize(),
    #                     "compsim": key,
    #                     "layer": i,
    #                     "pearsoncor": compute_pearson_correlation(
    #                         torch.tensor(self.ds[key]), torch.tensor(self.ds["label_ok"])
    #                     ),
    #                     "spearman": compute_spearman_correlation(
    #                         torch.tensor(self.ds[key]), torch.tensor(self.ds["label_ok"])
    #                     )
    #                 })  
    #     return corrp      

    # def run(self):
    #     num_layers = self.number_of_layers + 1
    #     composition_functions = ["cls", "avg", "sum", "f_ind", "f_joint", "f_inf"]
    #     similarity_metrics = ["Cos", "Euc", "Icm"]

    #     print("Preparing embeddings data...")
    #     results, embeddings_data = self.compute_embeddings(num_layers,composition_functions)

    #     print("Normalizing and aligning to similarity...")
    #     normalized_results = self.normalize_and_align_similarity(results, num_layers, composition_functions)

    #     self.ds = pd.DataFrame(normalized_results)
    #     print("Saving data to CSV...")
    #     self.save_to_csv(self.ds, "metrics")
    #     print("Saving embeddings...")
    #     self.save_to_disk(embeddings_data, "embeddings")

    #     print("Computing correlations...")
    #     corrp = self.compute_correlations(num_layers, composition_functions, similarity_metrics)

    #     print("Saving correlation data...")
    #     corrs = pd.DataFrame(corrp)
    #     self.save_to_csv(corrs, "correlation")

    def run(self):

        print("Preparing embeddings data...")
        # embeddings_data = []
        def extract_embeddings_compute_similarity_and_save(x):
            idx = x["idx"]
            s1 = x["sentence1"]
            s2 = x["sentence2"]
            base_embedding_1 = self.model.get_embeddings([s1])
            base_embedding_2 = self.model.get_embeddings([s2])
            
            composition_functions = ["cls", "avg", "sum", "f_ind", "f_joint", "f_inf"]
            for compf in composition_functions:
                for i in range(0, self.number_of_layers+1):                    
                    e1 = self.model.get_embedding_from_layer_with_composition_function(base_embedding_1, compf, i)
                    e2 = self.model.get_embedding_from_layer_with_composition_function(base_embedding_2, compf, i)

                    x["Cos"+compf.capitalize()+str(i)] = compute_cosine_similarity(e1, e2)
                    x["Euc"+compf.capitalize()+str(i)] = compute_euclidean_distance(e1, e2)
                    x["Icm"+compf.capitalize()+str(i)] = compute_icmb(e1, e2)

                    # embeddings_data.append({"idx": idx, 
                    #                         "model": self.model_name, 
                    #                         "compf":compf, 
                    #                         "layer":i, 
                    #                         "sentence1": e1, 
                    #                         "sentence2": e2})

            return x

        self.ds=self.ds.apply(extract_embeddings_compute_similarity_and_save, axis=1)
        
        print("Normalizing and aligning to similarity...")
        scaler = MinMaxScaler()
        composition_functions = ["cls", "avg", "sum", "f_ind", "f_joint", "f_inf"]
        for compf in composition_functions:
            for i in range(0, self.number_of_layers+1): 
                # Normalize euclidean distance and ICM
                self.ds[['Euc'+compf.capitalize()+str(i), 'Icm'+compf.capitalize()+str(i)]] = scaler.fit_transform(self.ds[['Euc'+compf.capitalize()+str(i), 'Icm'+compf.capitalize()+str(i)]])

                # Invert the normalized euclidean distance to align it to similarity
                self.ds['Euc'+compf.capitalize()+str(i)] = 1 - self.ds['Euc'+compf.capitalize()+str(i)]

        self.ds["model"] = self.model_name
        print("Saving data to CSV...")
        self.save_to_csv(self.ds, "metrics")

        # print("Saving embeddings...")
        # self.save_to_disk(embeddings_data, "embeddings") 

        corrp = []
        print("Computing correlations...")
        composition_functions = ["cls", "avg", "sum", "f_ind", "f_joint", "f_inf"]
        similarity_functions = ["Cos", "Euc", "Icm"]
        for simf in similarity_functions:  
            for compf in composition_functions:
                # For every single layer in the model
                for i in range(0, self.number_of_layers+1):
                    print(simf+compf.capitalize()+str(i),type(self.ds[simf+compf.capitalize()+str(i)]))
                    corrp.append(
                        {
                            "model": self.model_name,
                            "type": "Transformer",
                            "composition": simf,
                            "similarity": compf.capitalize(),                        
                            "compsim": simf+compf.capitalize(),
                            "layer": str(i),
                            "pearsoncor": compute_pearson_correlation(
                                torch.tensor(self.ds[simf+compf.capitalize()+str(i)]), 
                                torch.tensor(self.ds["label_ok"])
                            ),
                            "spearman": compute_spearman_correlation(
                                torch.tensor(self.ds[simf+compf.capitalize()+str(i)]), 
                                torch.tensor(self.ds["label_ok"])
                            )
                        }
                    )

        print("Saving correlation data...")
        corrs = pd.DataFrame(corrp)
        self.save_to_csv(corrs, "correlation")


class ExperimentStaticModel(Experiment):
    def __init__(self, model_name):
        self.expdir = "exptfm"
        self.results = "results"
        self.model_name = model_name
        print("Loading static embedding model '"+self.model_name+"'...")
        self.model = StaticEmbeddingModel(self.model_name)
        print("Loading dataset...")
        self.ds = load_dataset_as_pandas("stsb", "validation")
        self.embeddings = []
    
    def run(self):
        print("Preparing embeddings data...")

        composition_functions = ["cls", "avg", "sum", "f_ind", "f_joint", "f_inf"]

        # embeddings_data = []
        def extract_embeddings_compute_similarity_and_save(x):
            for compf in composition_functions:
                e = self.model.encode_with_composition_function([ x["sentence1"], x["sentence2"] ], compf)
                x["Cos"+compf.capitalize()] = compute_cosine_similarity(e[0].unsqueeze(0), e[1].unsqueeze(0))
                x["Euc"+compf.capitalize()] = compute_euclidean_distance(e[0].unsqueeze(0), e[1].unsqueeze(0))
                x["Icm"+compf.capitalize()] = compute_icmb(e[0].unsqueeze(0), e[1].unsqueeze(0))
                
                # embeddings_data.append({"idx": x["idx"], 
                #                         "model": self.model_name, 
                #                         "compf":compf, 
                #                         "sentence1": e[0], 
                #                         "sentence2": e[1]})

            return x

        self.ds = self.ds.apply(extract_embeddings_compute_similarity_and_save, axis=1)

        scaler = MinMaxScaler()

        for compf in composition_functions:
            # Normalize euclidean distance and ICM
            self.ds[['Euc'+compf.capitalize(), 'Icm'+compf.capitalize()]] = scaler.fit_transform(self.ds[['Euc'+compf.capitalize(), 'Icm'+compf.capitalize()]])

            # Invert the normalized euclidean distance to align it to similarity
            self.ds['Euc'+compf.capitalize()] = 1 - self.ds['Euc'+compf.capitalize()]

        self.ds["model"] = self.model_name
        print("Saving data to CSV...")
        self.save_to_csv(self.ds, "metrics")

        # print("Saving embeddings...")
        # self.save_to_disk(embeddings_data, "embeddings")   


        corrp = []
        print("Computing correlations...")
        similarity_functions = ["Cos", "Euc", "Icm"]
        composition_functions = ["cls", "avg", "sum", "f_ind", "f_joint", "f_inf"]
        for simf in similarity_functions:        
            for compf in composition_functions:
                corrp.append(
                    {
                        "model": self.model_name,
                        "type": "Static",
                        "composition": simf,
                        "similarity": compf.capitalize(),
                        "compsim": simf+compf.capitalize(),
                        "layer": "1",
                        "pearsoncor": compute_pearson_correlation(
                            torch.tensor(self.ds[simf+compf.capitalize()]), 
                            torch.tensor(self.ds["label_ok"])
                        ),
                        "spearman": compute_spearman_correlation(
                            torch.tensor(self.ds[simf+compf.capitalize()]), 
                            torch.tensor(self.ds["label_ok"])
                        )
                    }
                )
        
        print("Saving correlation data...")
        corrs = pd.DataFrame(corrp)
        self.save_to_csv(corrs, "correlation")
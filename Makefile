##############################################################################
# Global Variables
##############################################################################

# hyper parameters
ALPHABET = DNA
EPOCHS = 100

# number of greengene sequences to use for train, val test split of data
# values ensure a roughly 80% train, 10% val, 10% test split
MULTIPLICITY = 11
TRAIN = 7000 # 7000 * multiplicity samples
VAL = 100 # 100 * 100 samples
TEST = 150 # 150 * 150 samples
REF = 50000
QUERY = 500

# directory names
RAW_DIR = data/raw
INTERIM_DIR = data/interim
PROCESSED_DIR = data/processed
MODELS_DIR = models
GREENGENES_EMBEDDINGS_DIR = data/processed/greengenes_embeddings
OTU_EMBEDDINGS_DIR = data/processed/otu_embeddings
PROCESSED_GREENGENES_DIR = data/processed/greengenes

# file names
PROCESSED_GREENGENES_FILES = $(INTERIM_DIR)/greengenes/sequences_distances.pickle $(INTERIM_DIR)/greengenes/closest_strings.pickle $(INTERIM_DIR)/greengenes/auxillary_data.pickle
PROCESSED_IHMP_FILES = $(INTERIM_DIR)/ihmp/ibd_data.pickle \
					   $(INTERIM_DIR)/ihmp/ibd_metadata.pickle \
					   $(INTERIM_DIR)/ihmp/t2d_data.pickle \
					   $(INTERIM_DIR)/ihmp/t2d_metadata.pickle \
					   $(INTERIM_DIR)/ihmp/moms_data.pickle \
					   $(INTERIM_DIR)/ihmp/moms_metadata.pickle

TRANSFORMER_H16_MODEL = $(MODELS_DIR)/transformer_hyperbolic_16_model.pickle 
TRANSFORMER_E16_MODEL = $(MODELS_DIR)/transformer_euclidean_16_model.pickle
TRANSFORMER_H128_MODEL = $(MODELS_DIR)/transformer_hyperbolic_128_model.pickle
TRANSFORMER_E128_MODEL = $(MODELS_DIR)/transformer_euclidean_128_model.pickle

CNN_H2_MODEL = $(MODELS_DIR)/cnn_hyperbolic_2_model.pickle
CNN_H4_MODEL = $(MODELS_DIR)/cnn_hyperbolic_4_model.pickle
CNN_H6_MODEL = $(MODELS_DIR)/cnn_hyperbolic_6_model.pickle
CNN_H8_MODEL = $(MODELS_DIR)/cnn_hyperbolic_8_model.pickle
CNN_E2_MODEL = $(MODELS_DIR)/cnn_euclidean_2_model.pickle
CNN_E4_MODEL = $(MODELS_DIR)/cnn_euclidean_4_model.pickle
CNN_E6_MODEL = $(MODELS_DIR)/cnn_euclidean_6_model.pickle
CNN_E8_MODEL = $(MODELS_DIR)/cnn_euclidean_8_model.pickle

CNN_H16_MODEL = $(MODELS_DIR)/cnn_hyperbolic_16_model.pickle
CNN_E16_MODEL = $(MODELS_DIR)/cnn_euclidean_16_model.pickle
CNN_H128_MODEL = $(MODELS_DIR)/cnn_hyperbolic_128_model.pickle
CNN_E128_MODEL = $(MODELS_DIR)/cnn_euclidean_128_model.pickle

TRANSFORMER_H16_GREENGENES_EMBEDDINGS = $(GREENGENES_EMBEDDINGS_DIR)/transformer_hyperbolic_16_greengenes_embeddings.pickle
TRANSFORMER_H16_OTU_EMBEDDINGS = $(OTU_EMBEDDINGS_DIR)/transformer_hyperbolic_16_otu_embeddings.pickle

# configurations
PYTHON_INTERPRETER = python
.PHONY = all clean \
		 download_greengenes download_moms_pi \
		 process_greengenes process_ihmp \
		 train_transformer train_transformers train_cnn train_cnns \
		 greengenes_embeddings \
		 feedforward_otu_embeddings transformer_otu_embeddings \
		 feedforward_mixture_embeddings

##############################################################################
# Commands (Phony Targets)
##############################################################################


otu_embeddings: $(TRANSFORMER_H16_OTU_EMBEDDINGS)
greengenes_embeddings: $(TRANSFORMER_H16_GREENGENES_EMBEDDINGS)

train_cnn: $(CNN_H16_MODEL)
train_transformer: $(TRANSFORMER_E16_MODEL)
train_cnns: $(CNN_H16_MODEL) $(CNN_H128_MODEL) $(CNN_E16_MODEL) $(CNN_E128_MODEL)
# train_cnns: $(CNN_E2_MODEL) $(CNN_E4_MODEL) $(CNN_E6_MODEL) $(CNN_E8_MODEL) $(CNN_H2_MODEL) $(CNN_H4_MODEL) $(CNN_H6_MODEL) $(CNN_H8_MODEL)
train_transformers: $(TRANSFORMER_H16_MODEL) $(TRANSFORMER_H128_MODEL) $(TRANSFORMER_E16_MODEL) $(TRANSFORMER_E128_MODEL)

process_greengenes: $(PROCESSED_GREENGENES_FILES)
process_ihmp: $(PROCESSED_IHMP_FILES)

download_greengenes: $(RAW_DIR)/greengenes/gg_13_5.fasta

##############################################################################
# Get Mixture Embeddings
##############################################################################

mixture_embeddings: src/data/otu_to_mixture_embeddings.py src/data/parse_moms_pi.py
	$(PYTHON_INTERPRETER) $< \
		--otu_tables_path 'data/interim/moms_pi/16s_tables.pkl' \
		--id_to_embedding_path 'data/processed/greengenes/mlpencoder_id_to_embedding.pickle' \
		--model_path 'models/MLPEncoder.pickle' \
		--sample_data_to_sample_id_path 'data/interim/moms_pi/sample_data_to_sample_id.pickle' \
		--out_path 'data/processed/greengenes/sample_id_to_mixture_embedding.pickle'

##############################################################################
# Get OTU Embeddings
##############################################################################

$(TRANSFORMER_H16_OTU_EMBEDDINGS): src/embeddings/otu_embeddings.py $(TRANSFORMER_H16_GREENGENES_EMBEDDINGS)
	$(PYTHON_INTERPRETER) $< \
		--outdir $(OTU_EMBEDDINGS_DIR) \
		--ihmp_data $(INTERIM_DIR)/ihmp \
		--greengenes_embeddings $(TRANSFORMER_H16_GREENGENES_EMBEDDINGS)
	@touch $(TRANSFORMER_H16_OTU_EMBEDDINGS)


##############################################################################
# Greengenes Embeddings
##############################################################################

$(TRANSFORMER_H16_GREENGENES_EMBEDDINGS): src/embeddings/greengenes_embeddings.py $(TRANSFORMER_H16_MODEL)
	$(PYTHON_INTERPRETER) $< \
		--outdir $(PROCESSED_DIR)/greengenes_embeddings \
		--model $(TRANSFORMER_H16_MODEL) \
		--aux_data $(INTERIM_DIR)/greengenes/auxillary_data.pickle

##############################################################################
# Train CNN
##############################################################################

$(CNN_H2_MODEL): src/models/cnn/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=2 \
		--distance=hyperbolic --scaling=True  \
		--loss=mse \
		--batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size=128 \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(CNN_H2_MODEL)

$(CNN_H4_MODEL): src/models/cnn/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=4 \
		--distance=hyperbolic --scaling=True  \
		--loss=mse \
		--batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size=128 \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(CNN_H4_MODEL)

$(CNN_H6_MODEL): src/models/cnn/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=6 \
		--distance=hyperbolic --scaling=True  \
		--loss=mse \
		--batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size=128 \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(CNN_H6_MODEL)

$(CNN_H8_MODEL): src/models/cnn/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=8 \
		--distance=hyperbolic --scaling=True  \
		--loss=mse \
		--batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size=128 \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(CNN_H8_MODEL)

$(CNN_H16_MODEL): src/models/cnn/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=16 \
		--distance=hyperbolic --scaling=True  \
		--loss=mse \
		--batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size=128 \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(CNN_H16_MODEL)

$(CNN_H128_MODEL): src/models/cnn/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=128 \
		--distance=hyperbolic --scaling=True  \
		--loss=mse \
		--batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size=128 \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(CNN_H128_MODEL)

$(CNN_E2_MODEL): src/models/cnn/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=2 \
		--distance=euclidean  \
		--loss=mse \
		--batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size=128 \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(CNN_E2_MODEL)

$(CNN_E4_MODEL): src/models/cnn/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=4 \
		--distance=euclidean  \
		--loss=mse \
		--batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size=128 \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(CNN_E4_MODEL)

$(CNN_E6_MODEL): src/models/cnn/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=6 \
		--distance=euclidean  \
		--loss=mse \
		--batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size=128 \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(CNN_E6_MODEL)

$(CNN_E8_MODEL): src/models/cnn/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=8 \
		--distance=euclidean  \
		--loss=mse \
		--batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size=128 \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(CNN_E8_MODEL)

$(CNN_E16_MODEL): src/models/cnn/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=16 \
		--distance=euclidean  \
		--loss=mse \
		--batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size=128 \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(CNN_E16_MODEL)

$(CNN_E128_MODEL): src/models/cnn/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=128 \
		--distance=euclidean  \
		--loss=mse \
		--batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size=128 \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(CNN_E128_MODEL)


##############################################################################
# Train (Global) Transformers
##############################################################################

$(TRANSFORMER_H16_MODEL): src/models/transformer/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py # src/models/transformer/model.py src/models/task/dataset.py src/models/hyperbolics.py src/util/data_handling/data_loader.py src/util/ml_and_math/loss_functions.py src/visualization/visualize.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
 		--embedding_size=16 \
		--distance=hyperbolic --scaling=True \
		--segment_size=64 --heads=2 --trans_layers=2 --hidden_size=16 --layer_norm=True --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --layer_norm=True --batch_size=128 \
		--loss=mse \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity=$(MULTIPLICITY) \
 		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(TRANSFORMER_H16_MODEL)

$(TRANSFORMER_H128_MODEL): src/models/transformer/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py # src/models/transformer/model.py src/models/task/dataset.py src/models/hyperbolics.py src/util/data_handling/data_loader.py src/util/ml_and_math/loss_functions.py src/visualization/visualize.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=128 \
		--distance=hyperbolic --scaling=True \
		--segment_size=64 --heads=2 --trans_layers=2 --hidden_size=16 --layer_norm=True --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --layer_norm=True --batch_size=128 \
		--loss=mse \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity $(MULTIPLICITY) \
		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(TRANSFORMER_H128_MODEL)

$(TRANSFORMER_E16_MODEL): src/models/transformer/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py # src/models/transformer/model.py src/models/task/dataset.py src/models/hyperbolics.py src/util/data_handling/data_loader.py src/util/ml_and_math/loss_functions.py src/visualization/visualize.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=16 \
		--distance=euclidean \
		--segment_size=64 --heads=2 --trans_layers=2 --hidden_size=16 --layer_norm=True --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --layer_norm=True --batch_size=128 \
		--loss=mse \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity $(MULTIPLICITY) \
		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(TRANSFORMER_E16_MODEL)

$(TRANSFORMER_E128_MODEL): src/models/transformer/train.py src/models/train.py $(INTERIM_DIR)/greengenes/sequences_distances.pickle src/models/pair_encoder.py # src/models/transformer/model.py src/models/task/dataset.py src/models/hyperbolics.py src/util/data_handling/data_loader.py src/util/ml_and_math/loss_functions.py src/visualization/visualize.py
	$(PYTHON_INTERPRETER) $< \
		--epochs=$(EPOCHS) \
		--embedding_size=128 \
		--distance=euclidean \
		--segment_size=64 --heads=2 --trans_layers=2 --hidden_size=16 --layer_norm=True --readout_layers=1 \
		--lr=0.001 --weight_decay=0.0 --dropout=0.0 --layer_norm=True --batch_size=128 \
		--loss=mse \
		--data=$(INTERIM_DIR)/greengenes/sequences_distances.pickle --multiplicity $(MULTIPLICITY) \
		--out=$(MODELS_DIR) \
		--print_every=5 --patience=50 \
		--plot --save --use_wandb
	@touch $(TRANSFORMER_E128_MODEL)


##############################################################################
# Process Data 
##############################################################################

# process ihmp datasets
$(PROCESSED_IHMP_FILES): src/data/process_ihmp.py src/util/data_handling/data_loader.py
	$(PYTHON_INTERPRETER) $< \
		--ihmp_dir $(RAW_DIR)/ihmp \
		--outdir $(INTERIM_DIR)/ihmp
	@touch $(PROCESSED_IHMP_FILES)


# process greengenes fasta for traning models on edit distance 
$(PROCESSED_GREENGENES_FILES): src/data/process_fasta.py src/data/edit_distance.py src/util/data_handling/data_loader.py $(RAW_DIR)/greengenes/gg_13_5.fasta
	$(PYTHON_INTERPRETER) $< \
		--source_sequences $(RAW_DIR)/greengenes/gg_13_5.fasta \
		--out $(INTERIM_DIR)/greengenes \
		--alphabet $(ALPHABET) \
		--train_size $(TRAIN) --val_size $(VAL) --test_size $(TEST) \
		--ref_size $(REF) --query_size $(QUERY) \
		--compute_eda False --compute_csr False
	@touch $(PROCESSED_GREENGENES_FILES)

##############################################################################
# Download Data
##############################################################################

# download greengeens
$(RAW_DIR)/greengenes/gg_13_5.fasta: src/data/download_greengenes.sh
	bash $<
	@touch $@

##############################################################################
# Utilities
##############################################################################

clean:
	rm -rf data/interim/greengenes
	rm -rf data/processed/greengenes
	# keep models and figures for now while experimenting because they take a long time to train
	# rm -rf models
	# rm -rf figures
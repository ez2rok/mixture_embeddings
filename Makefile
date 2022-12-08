##############################################################################
# Global Variables
##############################################################################

# hyper parameters
ALPHABET = DNA
TRAIN = 7000
VAL = 1500
TEST = 1500

# directory names
RAW_GREENGENES_DIR = data/raw/greengenes/
INTERIM_MOMS_PI_DIR = data/interim/moms_pi/
INTERIM_GREENGENES_DIR = data/interim/greengenes/
PROCESSED_GREENGENES_DIR = data/processed/greengenes/
RAW_MOMS_PI_DIR = data/raw/moms_pi/
MODELS_DIR = models/

# file names
PARSED_GREENGENE_FILES = $(INTERIM_GREENGENES_DIR)sequences_distances.pickle \
						 $(INTERIM_GREENGENES_DIR)auxillary_data.pickle
PARSED_MOMS_PI_FILES = $(INTERIM_MOMS_PI_DIR)16s_tables.pkl \
					   $(INTERIM_MOMS_PI_DIR)otu_tables_normed_and_cleaned.pickle \
					   $(INTERIM_MOMS_PI_DIR)sample_data_to_sample_id.pickle
PROCESSED_GREENGENES_FILES = $(PROCESSED_GREENGENES_DIR)id_to_sequence_embedding.pickle

MLP_MODEL = models/MLPEncoder.pickle
TRANSFORMER_MODEL = models/Transformer.pickle 


# configurations
PYTHON_INTERPRETER = python
.PHONY = all clean \
		 download_greengenes download_moms_pi \
		 parse_greengenes parse_moms_pi \
		 train_transformer train_feedforward train_all \
		 feedforward_otu_embeddings transformer_otu_embeddings \
		 feedforward_mixture_embeddings \


##############################################################################
# Commands (Phony Targets)
##############################################################################

feedforward_mixture_embeddings: src/data/otu_to_mixture_embeddings.py src/data/parse_moms_pi.py
transformer_mixtures_embeddings: src/data/otu_to_mixture_embeddings.py src/data/parse_moms_pi.py

get_feedforward_otu_embeddings: $(PROCESSED_GREENGENES_DIR)mlpencoder_id_to_embedding.pickle
get_transformer_otu_embeddings: $(PROCESSED_GREENGENES_DIR)transformer_id_to_embedding.pickle # not implemented

train_all: train_transformer train_feedforward
train_transformer: $(TRANSFORMER_MODEL)
train_feedforward: $(MLP_MODEL)

parse_greengenes: $(PARSED_GREENGENE_FILES)
parse_moms_pi: $(PARSED_MOMS_PI_FILES)

download_greengenes: $(RAW_GREENGENES_DIR)gg_13_5.fasta
download_moms_pi: $(RAW_MOMS_PI_DIR)moms_pi

##############################################################################
# Get Mixture Embeddings
##############################################################################

mixture_embeddings: src/data/otu_to_mixture_embeddings.py src/data/parse_moms_pi.py
	$(PYTHON_INTERPRETER) $< \
		--moms_pi_tables_path 'data/interim/moms_pi/16s_tables.pkl' \
		--id_to_embedding_path 'data/processed/greengenes/mlpencoder_id_to_embedding.pickle' \
		--model_path 'models/MLPEncoder.pickle' \
		--sample_data_to_sample_id_path 'data/interim/moms_pi/sample_data_to_sample_id.pickle' \
		--out_path 'data/processed/greengenes/sample_id_to_mixture_embedding.pickle'

##############################################################################
# Get OTU Embeddings
##############################################################################

# map sequences to feedforward embeddings
$(PROCESSED_GREENGENES_DIR)mlpencoder_id_to_embedding.pickle: src/models/sequence_to_embeddings.py $(PARSED_GREENGENE_FILES) $(MLP_MODEL)
	$(PYTHON_INTERPRETER) $< \
		--out $(PROCESSED_GREENGENES_DIR) \
		--aux_data $(INTERIM_GREENGENES_DIR)/auxillary_data.pickle \
		--model $(MLP_MODEL)
	@touch $@

##############################################################################
# Train Models
##############################################################################

# train mlp model
$(MLP_MODEL): src/models/feedforward/train.py src/models/train.py $(INTERIM_GREENGENES_DIR)sequences_distances.pickle src/models/pair_encoder.py src/visualization/visualize.py # src/models/transformer/model.py src/models/task/dataset.py src/models/hyperbolics.py util/data_handling/data_loader.py util/ml_and_math/loss_functions.py 
	$(PYTHON_INTERPRETER) $< \
		--data=$(INTERIM_GREENGENES_DIR)sequences_distances.pickle \
		--out=$(MODELS_DIR) \
		--scaling=True --loss=mse --distance=hyperbolic \
		--batch_norm=True --lr=0.01 --weight_decay=0.00001 \
		--dropout=0.0 --embedding_size=128 --hidden_size=256 \
		--layer=3 --print_every=5 --patience=50 --epochs=500 \
		--batch_size=128 --plot
	@touch $(MLP_MODEL)

# train transformer model
$(TRANSFORMER_MODEL): src/models/transformer/train.py src/models/train.py $(INTERIM_GREENGENES_DIR)sequences_distances.pickle src/models/pair_encoder.py # src/models/transformer/model.py src/models/task/dataset.py src/models/hyperbolics.py util/data_handling/data_loader.py util/ml_and_math/loss_functions.py src/visualization/visualize.py
	$(PYTHON_INTERPRETER) $< \
		--data=$(INTERIM_GREENGENES_DIR)sequences_distances.pickle \
		--out=$(MODELS_DIR) \
		--scaling=True --loss=mse --distance=hyperbolic \
		--lr=0.01 --weight_decay=0.0 --layer_norm=True \
		--dropout=0.0 --embedding_size=128 --hidden_size=256 \
		--print_every=5 --patience=50 --epochs=500 \
		--batch_size=128 --plot
	@touch $(TRANSFORMER_MODEL)

##############################################################################
# Parse Data 
##############################################################################

# parse moms pi
$(PARSED_MOMS_PI_FILES): src/data/parse_moms_pi.py util/data_handling/data_loader.py
	$(PYTHON_INTERPRETER) $<
	@touch $(PARSED_MOMS_PI_FILES)

# parse greengenes
$(PARSED_GREENGENE_FILES): src/data/parse_fasta.py src/data/edit_distance.py util/alphabets.py util/data_handling/data_loader.py $(RAW_GREENGENES_DIR)gg_13_5.fasta
	$(PYTHON_INTERPRETER) $< \
		--input $(RAW_GREENGENES_DIR)gg_13_5.fasta \
		--out $(INTERIM_GREENGENES_DIR) \
		--alphabet $(ALPHABET) \
		--train_size $(TRAIN) --val_size $(VAL) --test_size $(TEST)
	@touch $(PARSED_GREENGENE_FILES)

##############################################################################
# Download Data
##############################################################################

# download greengeens
$(RAW_GREENGENES_DIR)gg_13_5.fasta: src/data/download_greengenes.sh
	bash $<
	@touch $@

# download moms_pi
# this does not work because the file is not created
$(RAW_MOMS_PI_DIR)moms_pi: src/data/download_moms_pi.sh
	bash $<
	@touch $@

##############################################################################
# Utilities
##############################################################################

clean:
	rm -rf data/raw/greengenes
	rm -rf data/interim/greengenes
	rm -rf data/processed/greengenes
	# keep models and figures for now while experimenting because they take a long time to train
	# rm -rf models
	# rm -rf figures
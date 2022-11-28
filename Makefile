##############################################################################
# Global Variables
##############################################################################

# hyper parameters
ALPHABET = DNA
TRAIN = 150
VAL = 150
TEST = 150


# directory names
INTERIM_GREENGENES_DIR = data/interim/greengenes/
PROCESSED_GREENGENES_DIR = data/processed/greengenes/
MODELS_DIR = models/


# file names
RAW_GREENGENES_FILE = data/raw/greengenes/gg_13_5.fasta
RAW_MOMS_PI_FILE = data/raw/moms_pi

INTERIM_GREENGENES_FILES = $(INTERIM_GREENGENES_DIR)sequences_distances.pickle \
						   $(INTERIM_GREENGENES_DIR)/auxillary_data.pickle

PROCESSED_GREENGENES_FILES = $(PROCESSED_GREENGENES_DIR)id_to_sequence_embedding.pickle

FEEDFORWARD_MODEL = models/MLPEncoder.pickle
TRANSFORMER_MODEL = models/Transformer.pickle 


# configurations
PYTHON_INTERPRETER = python
.PHONY = all clean


#################################################################################
# Commands
#################################################################################

mixture_embeddings: src/data/make_mixture_embeddings.py src/data/parse_moms_pi.py
	$(PYTHON_INTERPRETER) $< \
		--moms_pi_tables_path 'data/interim/moms_pi/16s_tables.pkl' \
		--id_to_embedding_path 'data/processed/greengenes/mlpencoder_id_to_embedding.pickle' \
		--model_path 'models/MLPEncoder.pickle' \
		--sample_data_to_sample_id_path 'data/interim/moms_pi/sample_data_to_sample_id.pickle' \
		--out_path 'data/processed/greengenes/mixture_embeddings.pickle'

models: $(FEEDFORWARD_MODEL) $(TRANSFORMER_MODEL) $(PROCESSED_GREENGENES_DIR)mlpencoder_id_to_embedding.pickle


############ Models ############

# map sequences to feedforward embeddings
$(PROCESSED_GREENGENES_DIR)mlpencoder_id_to_embedding.pickle: src/models/sequence_to_embeddings.py $(INTERIM_GREENGENES_FILES) $(FEEDFORWARD_MODEL)
	$(PYTHON_INTERPRETER) $< \
		--out $(PROCESSED_GREENGENES_DIR) \
		--aux_data $(INTERIM_GREENGENES_DIR)/auxillary_data.pickle \
		--model $(FEEDFORWARD_MODEL)
	@touch $@

# train feedforward model
$(FEEDFORWARD_MODEL): src/models/feedforward/train.py src/models/train.py $(INTERIM_GREENGENES_DIR)sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--data=$(INTERIM_GREENGENES_DIR)sequences_distances.pickle \
		--out=$(MODELS_DIR) \
		--scaling=True --loss=mse --distance=hyperbolic \
		--batch_norm=True --lr=0.01 --weight_decay=0.0 \
		--dropout=0.0 --embedding_size=128 --hidden_size=256 \
		--layer=3 --print_every=5 --patience=50 --epochs=500 \
		--batch_size=128
	@touch $(FEEDFORWARD_MODEL)

# train transformer model
$(TRANSFORMER_MODEL): src/models/transformer/train.py src/models/train.py $(INTERIM_GREENGENES_DIR)sequences_distances.pickle src/models/pair_encoder.py
	$(PYTHON_INTERPRETER) $< \
		--data=$(INTERIM_GREENGENES_DIR)sequences_distances.pickle \
		--out=$(MODELS_DIR) \
		--scaling=True --loss=mse --distance=hyperbolic \
		--lr=0.01 --weight_decay=0.0 --layer_norm=True \
		--dropout=0.0 --embedding_size=128 --hidden_size=256 \
		--print_every=5 --patience=50 --epochs=500 \
		--batch_size=128
	@touch $(TRANSFORMER_MODEL)


############ Data ############

# parse greengenes
$(INTERIM_GREENGENES_FILES): src/data/parse_fasta.py src/data/edit_distance.py util/alphabets.py util/data_handling/data_loader.py $(RAW_GREENGENES_FILE)
	$(PYTHON_INTERPRETER) $< \
		--input $(RAW_GREENGENES_FILE) \
		--out $(INTERIM_GREENGENES_DIR) \
		--alphabet $(ALPHABET) \
		--train_size $(TRAIN) --val_size $(VAL) --test_size $(TEST)
	@touch $(INTERIM_GREENGENES_FILES)

# download greengeens
$(RAW_GREENGENES_FILE): src/data/download_greengenes.sh
	bash $<
	@touch $@

# download moms_pi
# this does not work because the file is not created
$(RAW_MOMS_PI_FILE): src/data/download_moms_pi.sh
	bash $<
	@touch $@

clean:
	rm -rf data/raw/greengenes
	rm -rf data/interim/greengenes
	rm -rf data/processed/greengenes
	rm -rf models
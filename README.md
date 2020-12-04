# fMRI_encode

Steps to run:

1. Download the ABIDE dataset using the below command:
python download_abide_preproc.py -t NYU -p cpac -s filt_global -o NYU_dataset_fMRI -d func_preproc

2. This should download all NYU files in the NYU_dataset_fMRI directory.

3. To make a train-test split, run the generate_dataset.py file. This will create a file called train_test_split2.json that contains train-test split.

python generate_dataset.py


4. Run the following commands to create directories:
mkdir latent
mkdir reconstructed
mkdir models


4. To train a ML classifier, run the fmri_attention-e2e.py. Below are the functionalites:
	python fmri_attention-e2e.py train --> Start a new training session (You may need to create a new directory). The new model will be saved in the models/ directory
	
	python fmri_attention-e2e.py train models/model_folder/model.ckpt --> Restore this model and train from here.

	python fmri_attention-e2e.py viz models/model_folder/model.ckpt --> Take a random test image and produce reconstruction (with seq2seq and image). Images will be stored in the reconstructed directory (You may need to create this directory)

	python fmri_attention-e2e.py latent models/model_folder/model.ckpt ---> Generate latent representation (seq2seq final encoder state) and store in latent directory (You may need to create this directory)

	python fmri_attention-e2e.py predict models/model_folder/model.ckpt ---> Get test case predictions. (use only if e2e training was done).


5. If only unsupervised training is required, just comment the below lines in the fmri_attention-e2e.py:

elif _ > 12000:
     training_mode = 'e2e'


6. To train supervised training classifier, use:
	First run: python fmri_attention-e2e.py latent models/model_folder/model.ckpt  -> This will generate latent representations
	Then run: python supervised_learning.py

7. To change the preprocessing and model type, change the below lines in supervised_learning.py

preprocessing_types = ['flat', 'mean_channels', 'pca', 'kernel_pca', 'nmf']
model_types = ['LR', 'RF', 'LDA', 'KNN', 'MLP']

8. To train CNN model on latent space seperately, use:

python supervised_training-CNN.py


Feel free to reach out to us in case of issues!!




	

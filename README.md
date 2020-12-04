# fMRI_encode

CPSC 8420: Advanced Machine Learning

Avishek Biswas (avisheb@clemson.edu)
Foram Joshi (fjoshi@clemson.edu)

### Running on Palmetto:

To run in palmetto, first get a compute node:<br>
qsub -I -l select=1:ncpus=28:mem=120gb:ngpus=1:gpu_model=p100,walltime=72:00:00

Load these modules:
module add anaconda3/5.1.0-gcc/8.3.1<br>
module load cudnn/7.6.5.32-10.2-linux-x64-gcc/8.3.1-cuda10_2<br>
module add openmpi/3.1.6-gcc/8.3.1-cuda10_2-ucx<br>
module add glew/2.0.0-gcc/8.3.1<br>
<br>
source activate <conda_env><br>


You need the below modules to run the code:
nibabel
tensorflow 1.15
numpy
pickle
json




### Steps to run:

1. Download the ABIDE dataset using the below command:
python download_abide_preproc.py -t NYU -p cpac -s filt_global -o NYU_dataset_fMRI -d func_preproc

2. This should download all NYU files in the NYU_dataset_fMRI directory.

3. To make a train-test split, run the generate_dataset.py file. This will create a file called train_test_split2.json that contains train-test split.

python generate_dataset.py


4. Run the following commands to create directories:
mkdir latent
mkdir reconstructed
mkdir models


5. To train a ML classifier, run the fmri_attention-e2e.py. Below are the functionalites: (Use palmetto to train.)
	python fmri_attention-e2e.py train --> Start a new training session (You may need to create a new directory). The new model will be saved in the models/ directory
	
	python fmri_attention-e2e.py train models/model_folder/model.ckpt --> Restore this model and train from here.

	python fmri_attention-e2e.py viz models/model_folder/model.ckpt --> Take a random test image and produce reconstruction (with seq2seq and image). Images will be stored in the reconstructed directory (You may need to create this directory)

	python fmri_attention-e2e.py latent models/model_folder/model.ckpt ---> Generate latent representation (seq2seq final encoder state) and store in latent directory (You may need to create this directory)

	python fmri_attention-e2e.py predict models/model_folder/model.ckpt ---> Get test case predictions. (use only if e2e training was done).


6. If only unsupervised training is required, just comment the below lines in the fmri_attention-e2e.py:

elif _ > 12000:
     training_mode = 'e2e'


7. To train supervised training classifier, use:
	First run: python fmri_attention-e2e.py latent models/model_folder/model.ckpt  -> This will generate latent representations
	Then run: python supervised_learning.py

8. To change the preprocessing and model type, change the below lines in supervised_learning.py

preprocessing_types = ['flat', 'mean_channels', 'pca', 'kernel_pca', 'nmf']
model_types = ['LR', 'RF', 'LDA', 'KNN', 'MLP']

9. To train CNN model on latent space seperately, use:

python supervised_training-CNN.py


Feel free to reach out to us in case of issues!!

	

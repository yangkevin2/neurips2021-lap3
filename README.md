# Learning Space Partitions for Path Planning

Code for Learning Space Partitions for Path Planning (http://arxiv.org/abs/2106.10544) by Kevin Yang, Tianjun Zhang, Chris Cummins, Brandon Cui, Benoit Steiner, Linnan Wang, Joseph E. Gonzalez, Dan Klein, and Yuandong Tian. 

Base LaMCTS code originally from https://github.com/facebookresearch/LaMCTS . 
MiniWorld and MiniGrid envs originally from https://github.com/maximecb/gym-miniworld and https://github.com/maximecb/gym-minigrid respectively.

# Setup

Required packages in `requirements.txt`. Additionally, install `pytorch` using `conda`; version 1.9.1 works fine. Install the `lamcts-planning` package via `setup.py` and also install the `latent-plan` directory. 

## MiniWorld and MiniGrid:

Install the `gym-miniworld` and/or `gym-minigrid` directories, which have our modifications from the original envs. **Make sure to install the ones included in this folder, not the official versions.** Otherwise you will see crashes. 

## Molecule:

Install the code for the latent space from https://github.com/wengong-jin/hgraph2graph/tree/e396dbaf43f9d4ac2ee2568a4d5f93ad9e78b767. Note this code assumes the use of a GPU. Our code that interfaces with the pretrained latent space assumes that you put the `hgraph2graph` code in the same top-level directory as the top-level `plalam` repository. Additionally, download our pretrained latent space ckpts by running `wget https://plalam-molecule-ckpts.s3.amazonaws.com/latent-molecule-ckpts.zip` and unzip the contents, putting them inside a `ckpt` folder inside the `hgraph2graph` repository (see the paths under `FakeArgs` in `lamcts_planning/util.py`). Finally, put `vocab.txt` inside the `hgraph2graph` repository at the top level. 

To setup the property evaluators, follow the instructions to install Chemprop at https://chemprop.readthedocs.io/en/latest/installation.html (in particular, make sure to get RDKit), and make sure to install version 1.2.0. Additionally, install `scikit-learn==0.21.3`. Then unzip the `molecule-ckpts.zip` file. There is no ckpt needed for QED since that's just a computationally defined property. `clf_py36.pkl` is the DRD2 property and should be placed in `lamcts_planning`. The other two folders are Chemprop ckpts for HIV and SARS, and the code currently assumes they exist in the same directory as the top-level `plalam` repository. 

NOTE: if you see `Warning: molecule dependencies not installed; install if running molecule exps` it means your setup is failing somewhere; see the imports at the top of `lamcts_planning/util.py`. 

# Example Commands: 

MiniWorld example: (see https://github.com/maximecb/gym-miniworld for discussion on why the `xvfb-run` stuff is needed when running on Linux without rendering; also follow their setup instructions for e.g. the OpenGL dependencies)

```
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -u main.py --env MiniWorld-MazeS3-v0 --num_threads 1 --num_trials 1 --method lamcts-planning --Cp 2 --horizon 216 --replan_freq 216 --iterations 2000 --final_obs_split --latent --latent_model cnn --latent_ckpt None --init_sigma_mult 8 
```

MiniGrid example: 

```
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -u main.py --env MiniGrid-DoorKey-6x6-v0 --num_threads 1 --num_trials 1 --method lamcts-planning --Cp 2 --horizon 180 --replan_freq 180 --iterations 2000 --final_obs_split --latent --latent_model cnn --latent_ckpt None
```

Molecule example: 

```
python -u -W ignore main.py --env QED --num_threads 1 --num_trials 1 --method lamcts-planning --Cp 0.5 --horizon 1 --replan_freq 1 --iterations 1000  --final_obs_split --action_seq_split --init_sigma_mult 1
```

Note the "horizon" is 1 here is just a code hack - we operate in a latent space which encodes the full action sequence. You might see a few `failed to decode` warnings, which is fine; the graph decoder we use fails on a small fraction of inputs. Just as long as it's not failing on all the inputs. 
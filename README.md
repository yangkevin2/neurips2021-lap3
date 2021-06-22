# Learning Space Partitions for Path Planning

Code for Learning Space Partitions for Path Planning (http://arxiv.org/abs/2106.10544) by Kevin Yang, Tianjun Zhang, Chris Cummins, Brandon Cui, Benoit Steiner, Linnan Wang, Joseph E. Gonzalez, Dan Klein, and Yuandong Tian. 

Base LaMCTS code originally from https://github.com/facebookresearch/LaMCTS . 
MiniWorld and MiniGrid envs originally from https://github.com/maximecb/gym-miniworld and https://github.com/maximecb/gym-minigrid respectively.

Required packages in `requirements.txt`, and also install https://github.com/wengong-jin/hgraph2graph/tree/e396dbaf43f9d4ac2ee2568a4d5f93ad9e78b767 if you want to run molecule exps. 
Install the `lamcts-planning` package via `setup.py` and also install the `latent-plan` directory. If you are running MiniWorld and/or MiniGrid, install the gym-minigrid and/or gym-miniworld directories, which have our modifications from the original envs. 

# Example Commands: 

MiniWorld example: (see https://github.com/maximecb/gym-miniworld for discussion on why the xvfb-run stuff is needed when running on Linux without rendering)

```
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -u main.py --env MiniWorld-MazeS3-v0 --num_threads 1 --num_trials 1 --method lamcts-planning --Cp 2 --horizon 216 --replan_freq 216 --iterations 2000 --final_obs_split --latent --latent_model cnn --latent_ckpt None --init_sigma_mult 8 
```

Molecule example: 

```
python -u -W ignore main.py --env QED --num_threads 1 --num_trials 1 --method lamcts-planning --Cp 0.5 --horizon 1 --replan_freq 1 --iterations 1000  --final_obs_split --action_seq_split --init_sigma_mult 1
```

Note the "horizon" is 1 here is just a code hack - we operate in a latent space which encodes the full action sequence. 
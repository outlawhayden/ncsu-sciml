#requires:
- datafiles
- dependent packages (should be standard via conda)
- potential backend config. should autodetect, but if not you might have to manually set jax.backend env variable

#files:

##dnn_optuna
- based on assignment model space, greedily search all permutations of model configuration based on objective function
- saves study information in memory as sqlite db, but this script also compresses and saves out loss trajectories for each trial, trial timing data, and optimal params. can run and walk away
- current number of studies is 20, but can vary

## dnn_optimal
- based on parameters from dnn_optuna (hardcoded), train NN to solve problem from datafile
- note i used custom eqx modules that override Linear, MLP classes. might not be compatible with preexisting frameworks i haven't seen. is entirely selfcontained minus dataset
- serializes weights out as `dnn_model_opt.eqx` for use in dnn_evaluate_optimal

## dnn_evaluate_optimal
- requires `dnn_model_opt.eqx` - reads in weights from file and initializes model class. note architecture must match configuration in ddn_optimal
- initializes model, and reconstructs entire solution surface over (x_1, x_2)
- renders heatmap of reconstructed surfaces. uses cubic interpolation to render smooth image, saves out as png

full code available at https://github.com/outlawhayden/ncsu-sciml/tree/main/hw2
    - does not include datafiles (*.npz, *.eqx)
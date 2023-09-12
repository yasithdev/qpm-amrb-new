import wandb

# Provide your entity and a project name when you
# use wandb.Api methods.
ENTITY_NAME = "yasith"
PROJECT_NAME = "robust_ml"
api = wandb.Api(overrides={"entity": ENTITY_NAME, "project": PROJECT_NAME})
runs = api.runs(ENTITY_NAME + "/" + PROJECT_NAME)

for run in runs:
    for artifact in run.logged_artifacts():
        # Clean up versions that don't have an alias such as 'latest'.
        # NOTE: You can put whatever deletion logic you want here.
        print(run.name, artifact.type, artifact.aliases)
        if len(artifact.aliases) == 0:
            artifact.delete()
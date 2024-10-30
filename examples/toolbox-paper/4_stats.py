import osl
import os
import numpy as np
import glmtools as glm
import matplotlib.pyplot as plt
from dask.distributed import Client

def first_level(dataset, userargs):
    DC = glm.design.DesignConfig()
    DC.add_regressor(name="famous", rtype="Categorical", codes=[5,6,7])
    DC.add_regressor(name="unfamiliar", rtype="Categorical", codes=[13,14,15])
    DC.add_regressor(name="scrambled", rtype="Categorical", codes=[17,18,19])
    DC.add_contrast(name="Mean", values={"famous": 1/3, "unfamiliar": 1/3, "scrambled": 1/3})
    DC.add_contrast(name="Faces - Scrambled", values={"famous": 1, "unfamiliar": 1, "scrambled": -2})
    dataset['glm'] = osl.glm.glm_epochs(DC, dataset['epochs'])
    dataset['glm'].design.plot_summary(savepath=os.path.join(
        os.path.dirname(dataset['raw'].filenames[0]), 'subject_design.png'))
    return dataset

def second_level(dataset, userargs):
    firstlevel_contrast = userargs.get('firstlevel_contrast', 'Faces - Scrambled')
    group_contrast = userargs.get('group_contrast', 'Mean')
    tmin = userargs.get('tmin', -np.Inf)
    tmax = userargs.get('tmax', np.Inf)
    
    groupDC = glm.design.DesignConfig()
    info = {"Subject": np.repeat(np.arange(1, 20), 6)}
    for i in range(19):
        # Add subject mean regressors
        groupDC.add_regressor(name=f"Subj{i+1}", rtype="Categorical", datainfo="Subject", codes=[i+1])    

    # add group contrast
    groupDC.add_contrast(name='Mean', values={f"Subj{i+1}": 1/19 for i in range(19)})
    
    # group level model
    dataset['group_glm'] = osl.glm.group_glm_epochs(dataset['glm'], groupDC)
    dataset['group_glm'].design.plot_summary(savepath=os.path.join(userargs['figdir'], 'group_design.png'))

    # max stat permutation test
    dataset['group_glm_perm'] = osl.glm.glm_base.SensorMaxStatPerm(dataset['group_glm'], dataset['group_glm'].contrast_names.index(group_contrast),
         dataset['group_glm'].fl_contrast_names.index(firstlevel_contrast), tmin=tmin, tmax=tmax)
    dataset['group_glm_perm'].plot_sig_clusters(99)
    plt.savefig(os.path.join(userargs['figdir'], f'group_contrast-{firstlevel_contrast}-significance.png'))
    return dataset
    

if __name__ == "__main__":
    client = Client(n_workers=16, threads_per_worker=1)
  
    config = """
      preproc:
        - read_dataset: {ftype: sflip_parc-raw}
        - epochs: {picks: misc, tmin: -0.2, tmax: 0.5}
        - first_level: {}
      group:
        - second_level: {tmin: 0.05, tmax: 0.3, figdir: ds117/figures}
    """
    
    proc_dir = "ds117/processed"
    src_files = sorted(osl.utils.Study(os.path.join(proc_dir, "sub{sub_id}-run{run_id}", "sub{sub_id}-run{run_id}_sflip_parc-raw.fif")).get())    
    subjects = [f"sub{i+1:03d}-run{j+1:02d}" for i in range(19) for j in range(6)]
 
    osl.preprocessing.run_proc_batch(
        config,
        src_files,
        subjects,
        outdir=proc_dir,
        ftype='raw',
        extra_funcs=[first_level, second_level],
        dask_client=True,
        overwrite=True,
        gen_report=False,
        skip_save=['events', 'raw', 'ica', 'event_id', 'sflip_parc-raw'],
        random_seed=3557485304,
    )


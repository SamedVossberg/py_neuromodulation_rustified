PATH_COORDS = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/out_dir/df_per_ind_all_coords.csv';
PATH_OUT = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/out_connectivity';

addpath("/Users/Timon/Documents/MATLAB/spm12");
addpath("/Users/Timon/Documents/MATLAB/leaddbs");
addpath("/Users/Timon/Documents/MATLAB/wjn_toolbox");

T=readtable(PATH_COORDS);

for a =1:size(T,1)
    roiname  = fullfile(PATH_OUT, strcat(string(T.sub(a)), '_ROI_', string(T.ch_orig{a}), '.nii'));
    mni = [abs(T.x(a)) T.y(a) T.z(a)];
    wjn_spherical_roi(roiname,mni,4);
end
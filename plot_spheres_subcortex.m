addpath("/Users/Timon/Documents/MATLAB/spm12");
addpath("/Users/Timon/Documents/MATLAB/leaddbs");
addpath("/Users/Timon/Documents/MATLAB/wjn_toolbox");
addpath("matlab_funcs")
PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/out_dir';
df = readtable(fullfile(PATH_PER, 'df_per_ind_all_coords.csv'), 'ReadRowNames', true);

fig = ea_mnifigure;

colors = colorlover(5);
% wjn_plot_surface(fullfile('meshes', 'cortex_bl.nii'), "#FF8000", 1);  % 0, 

% wjn_plot_surface(fullfile('meshes', 'STN_bl.nii'), "#FF8000", 1);  % 0, 
% wjn_plot_surface("'/Users/Timon/Documents/MATLAB/leaddbs/templates/space/MNI152NLin2009bAsym/atlases/DISTAL (Ewert 2017)/lh/GPi.nii'", "#FF8000", 0.3);  % 0, 

df_query = df((strcmp(df.loc, 'STN') | strcmp(df.loc, 'GP')) & strcmp(df.label, 'pkg_dk') & strcmp(df.classification, 'True'), :);
[contact_colours, colours_rgb] = map_values_to_cmap(df_query.per, 'viridis');

radius = 0.5;
[x, y, z] = sphere(100);
for row=1:height(df_query)
    %coords = str2num(df_query.ch_coords{row}) * 1000;
    ax_sphere = surf(df_query(row,:).x + (x.*radius), ...
                     df_query(row,:).y + (y.*radius), ...
                     df_query(row,:).z + (z.*radius));
    set(ax_sphere, 'LineStyle', 'none', 'facecolor', contact_colours(row, :), 'facealpha', 1);
end

ax_sphere = surf(12.58 + (x.*.7), ...
                 -13.41 + (y.*.7), ...
                 -5.87 + (z.*.7));
set(ax_sphere, 'LineStyle', 'none', 'facecolor', '#CA830F', 'facealpha', 1);

%%%%%%%%%% ECoG 
df_query = df(strcmp(df.loc, 'ECOG') & strcmp(df.label, 'pkg_dk') & strcmp(df.classification, 'True'), :);
[contact_colours, colours_rgb] = map_values_to_cmap(df_query.per, 'viridis');

radius = 1;
[x, y, z] = sphere(100);
for row=1:height(df_query)
    %coords = str2num(df_query.ch_coords{row}) * 1000;
    ax_sphere = surf(df_query(row,:).x + (x.*radius), ...
                     df_query(row,:).y + (y.*radius), ...
                     df_query(row,:).z + (z.*radius));
    set(ax_sphere, 'LineStyle', 'none', 'facecolor', contact_colours(row, :), 'facealpha', 1);
end

ax_sphere = surf(12.58 + (x.*.7), ...
                 -13.41 + (y.*.7), ...
                 -5.87 + (z.*.7));
set(ax_sphere, 'LineStyle', 'none', 'facecolor', '#CA830F', 'facealpha', 1);

cortex_mesh = load(fullfile('meshes/CortexHiRes.mat'));
plot_mesh.vertices = cortex_mesh.Vertices_rh;
plot_mesh.faces = cortex_mesh.Faces_rh;
[p,s,v] = wjn_plot_surface(plot_mesh,  "#FF8000", 0.);
alpha(p, 0.3)
fprintf("Hallo")
function gen_feb_file(folder_name, file_name_in, file_name_out, coors, mat)

%% This function creates a new .feb file for a cutometer simulation based on the set of coordinates and coefficients

fin = fopen(file_name_in,'r');
fout = fopen(sprintf('%s%s',folder_name,file_name_out),'w+');

s = fgets(fin);
while ~contains(s,'node')
    if contains(s,'skin')
        imat=1;
    elseif contains(s,'adipose')
        imat=2;
    end
    if contains(s,'<phi0>')
        s = strrep(s, '0.2', num2str(mat.phi(imat)));
    end
    if contains(s,'<E>')
        s = strrep(s, '10000', num2str(mat.E1(imat)));
    end
    if contains(s,'<v>')
        s = strrep(s, '0.4', num2str(mat.nu(imat)));
    end
    if contains(s,'<perm>')
        s = strrep(s, '5e-11', num2str(mat.perm1(imat)));
    end
    
    fprintf(fout,'%s',s);
    s = fgets(fin);
end

%% Read the coordinates
i=0;
while contains(s,'node')
    i=i+1;
    fprintf(fout,'\t\t\t<node id="%i">%14.12f,%14.12f,%14.12f</node>\n',i,coors(i,1),coors(i,2),coors(i,3));
    s = fgets(fin);
end


while ~feof(fin)
    fprintf(fout,'%s',s);
    s = fgets(fin);
end
fprintf(fout,'%s',s);

close('all')
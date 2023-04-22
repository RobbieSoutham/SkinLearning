function [] = read_nodal(iprobe, folder)
    display(pwd)
    str2 = sprintf('!mv %s/nodal_solution.txt %s/nodal_solution%i.txt',folder, folder, iprobe);
    display(str2)
    eval(str2)
    fin=fopen(sprintf('%s/nodal_solution%i.txt',folder, iprobe));
    k=0;
    while ~feof(fin)
        s = fgets(fin);
        if contains(s,'1559')
            sol(k,:) = textscan(s,'%d %f %f %f');
            s = fgets(fin);
        elseif contains(s,'Time')
            k=k+1;
            t(k)=str2num(s(9:end));
        end
    end
    
    sol=cell2mat(sol(:,2:4));
    sol = sol(:,3)-sol(1,3);
    data = [[0; t'],[0;sol]];
    writematrix(data,sprintf('%s/Disp%i.csv',folder,iprobe))
    
    % sol = csvread( 'Disp2.csv' )
    % plot(sol(:,3)-sol(1,3))
quit
end

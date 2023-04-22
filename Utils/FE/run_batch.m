

%% This file was written in order to run batch jobs for the cutometer simulations
% To do:
% - hypercube loop over the variables which have a range defined (based on literature) OR
% - monte carlo simulations
% - extend the original problem in FEBio to 2 layers
% - export stresses and extract those to get the response of the middle
% node
% - perform the parameter estimation
% - build a virtual database
% - speed up the simulation

function [] = run_batch(E1, nu, perm1, ttr, ttha, tths, exp)
    folder_name = ['SamplingResults2/' , exp, '/'];
    file_name_in = 'Cutometer_DoubleLayer_v2.feb';
    for iprobe = 1:2
        file_name_out = sprintf('Cutometer_out%i.feb',iprobe);
        mkdir(folder_name)
        
        %% material
        mat.E1 = [170e3 66e3];  %young's modulus [Pa]
        mat.E1 = E1;  %young's modulus [Pa]

        % mat.E2 = 4;
        mat.phi = [0.56 0.4];    % solid volume fraction [-]
        mat.phi = [0.1 0.1];    % solid volume fraction [-]
        mat.nu = [0.4 0.48];     % poisson ratio [-]
        mat.nu = nu;     % poisson ratio [-]
        
        % mat.perm1 = [5.5e-11 6e-11];  % permeability [Units??]
        mat.perm1 = [0.5e-11 4e-11];  % permeability [m^4/N.s]=[m^2/(Pa.s)]
        %     mat.perm1 = [0.5e-10 4e-10];  % permeability [m^4/N.s]=[m^2/(Pa.s)]
        mat.perm1 = perm1;
        
        % mat.perm2 = 8;
        
        %% geometry
        if iprobe == 1
            tpa = 1e-3;  % target radius of aperture [m] (2mm probe)
        elseif iprobe == 2
            tpa = 4e-3;  % target radius of aperture [m] (8mm probe)
        end


        
        %ttr = 5e-3;  % target radius of tissue sample [m]
        %ttha = 5e-3;  % target thickness for adipose tissues [m]     
        %tths = 1e-3;  % target thickness for skin tissue [m]

        H=mat.E1.*(1-mat.nu)./(1+mat.nu)./(1-2*mat.nu);
        K=mat.perm1;
        dz=[tths ttha];
        dt=dz.^2./(H.*K);
        
        
        fid_in = fopen(file_name_in,'r');
        coor_transform = coor_transformation(fid_in,tpa,ttr,ttha,tths);
        fclose('all');
        
        gen_feb_file(folder_name, file_name_in, file_name_out, coor_transform, mat);
        display(folder_name)
	    cd(folder_name)
        tic
        %
        %str1=sprintf('febio4 Cutometer_out%i.feb -silent',iprobe);
	%unix(str1)
	%display(str1)
	toc
        cd ..
	cd ..
        %% 
        %read_nodal(iprobe, folder_name)

    end

    quit
end



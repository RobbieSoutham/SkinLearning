function coor_transform = coor_transformation(fid,tpa,ttr,ttha,tths)

%% This function:
%- reads the coordinates
%- transforms them into polar coordinates
%- finds the nodes in the different mesh parts
%- linearly transforms the nodes of the different parts
%- passes back the new coordinates

%% some numbers for the template mesh
thickness.adipose = 1;
thickness.skin = 0.1;
probe_aperture = 1;    % 2mm radius
tissue_radius = 2;    % radius of tissue sample.

ths = thickness.skin;
tha = thickness.adipose;
pa = probe_aperture;
tr = tissue_radius;

% Skip first lines
s = fgets(fid);
strfind(s,'node')
while isempty(strfind(s,'node'))
    s = fgets(fid);
end

%% Read the coordinates
i=0;
while strfind(s,'node')
    i = i+1;
    x1 = strfind(s,'">');
    x2 = strfind(s,'</node');
    sub_s = s(x1+2:x2-1);
    coors(i,:) = str2num(sub_s);
    s = fgets(fid);
end

%% Transform the coordinates
[tt,rr] = cart2pol(coors(:,1),coors(:,2));
rnew = rr;
z = coors(:,3);

%% Identify nodes for 4 parts
nodes1 = find((coors(:,3)>=tha) & rr<=pa);   %  skin aperture
nodes2 = find((coors(:,3)<tha) & rr<=pa);   %  adipose aperture
nodes3 = find((coors(:,3)>=tha) & rr>pa);   %  skin probe
nodes4 = find((coors(:,3)<tha) & rr>pa);   %  adipose probe

%% perform radial transformations
% rnew = a r +b
% rnew (r=pa)=tpa   -> a=tpa/pa
% rnew(r=0) = 0    -> b=0

rnew(nodes1) = (tpa/pa)*rnew(nodes1);
rnew(nodes2) = (tpa/pa)*rnew(nodes2);

% rnew = a r +b
% rnew (r=pa)=tpa   -> tpa = a*pa + b  -> b=tpa - a *pa
% rnew(r=tr) = ttr    -> ttr = a*tr+b = a*tr+tpa-a*pa = a(tr-pa)+tpa
%                         a = (ttr-tpa)/(tr-pa)

a = (ttr-tpa)/(tr-pa);
b = tpa - a *pa;
rnew(nodes3) = a*rnew(nodes3)+b;
rnew(nodes4) = a*rnew(nodes4)+b;

%% perform vertical transformations

z(nodes2) = ttha/tha * z(nodes2);
z(nodes4) = ttha/tha * z(nodes4);

a = tths/ths;
b = ttha - a*tha;
z(nodes1) = a*z(nodes1)+b;
z(nodes3) = a*z(nodes3)+b;

%% Transform the coordinates back
[x,y] = pol2cart(tt,rnew);

coor_transform = [x,y,z];

% % % plot resulting transformation
% plot3(x(nodes1),y(nodes1),z(nodes1),'*r')
% hold on
% plot3(x(nodes2),y(nodes2),z(nodes2),'*m')
% plot3(x(nodes3),y(nodes3),z(nodes3),'*g')
% plot3(x(nodes4),y(nodes4),z(nodes4),'*y')
% % plot3(coors(:,1),coors(:,2),coors(:,3),'*b')
%
% %element 27
% figure
% plot3(x(22),y(22),z(22),'*k')
% hold on
% plot3(x(422),y(422),z(422),'*k')
% plot3(x(435),y(435),z(435),'*k')
% plot3(x(23),y(23),z(23),'*k')
% plot3(x(45),y(45),z(45),'*k')
% plot3(x(47),y(47),z(47),'*k')
% plot3(x(48),y(48),z(48),'*k')
% plot3(x(46),y(46),z(46),'*k')
%
% text(x(22),y(22),z(22),'22')
% text(x(422),y(422),z(422),'422')
% text(x(435),y(435),z(435),'435')
% text(x(23),y(23),z(23),'23')
% text(x(45),y(45),z(45),'45')
% text(x(47),y(47),z(47),'47')
% text(x(48),y(48),z(48),'48')
% text(x(46),y(46),z(46),'46')
% % axis equal
%
% x=coors(:,1)
% y=coors(:,2)
% z=coors(:,3)
% %element 27
% figure
% plot3(x(22),y(22),z(22),'*k')
% hold on
% plot3(x(422),y(422),z(422),'*k')
% plot3(x(435),y(435),z(435),'*k')
% plot3(x(23),y(23),z(23),'*k')
% plot3(x(45),y(45),z(45),'*k')
% plot3(x(47),y(47),z(47),'*k')
% plot3(x(48),y(48),z(48),'*k')
% plot3(x(46),y(46),z(46),'*k')
%
% text(x(22),y(22),z(22),'22')
% text(x(422),y(422),z(422),'422')
% text(x(435),y(435),z(435),'435')
% text(x(23),y(23),z(23),'23')
% text(x(45),y(45),z(45),'45')
% text(x(47),y(47),z(47),'47')
% text(x(48),y(48),z(48),'48')
% text(x(46),y(46),z(46),'46')
% % axis equal
%
%

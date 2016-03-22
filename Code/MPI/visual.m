clear all;
for i=0:300
    name=strcat('data/v',int2str(10000000+i),'.datbin');
    u=fopen(name);
    size=256;
    u1=fread(u,[size,size],'double');
    figure(11); clf;
    title(['Time ',num2str(i)]);
    surf(u1,'FaceColor','interp',...
   'EdgeColor','none',...
   'FaceLighting','phong');
end;
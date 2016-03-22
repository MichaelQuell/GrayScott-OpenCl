clear all;
c = {[0.4660    0.6740    0.1880]; [0.8500    0.3250    0.0980]; [0    0.4470    0.7410]; ...
  [0.9290    0.6940    0.1250]; [ 0.6350    0.0780    0.1840];[ 0.9    0.9    0.9];[ 0.9    0.0    0.9]};
%time in s
gt755m=[813.948975,180.307007,398.890991,1194.365967,5113.804199,29320.984375,154696.953125,1675567.000000]/1000;

apu7660=[630.838989,639.737000,934.677979,3312.325928,5976.629883,22480.279297,88562.703125,840320.687500]/1000;
radeonhd5450=[1459.677979,1175.531006,3839.747070,10815.346680,21306.988281,116964.515625,765088.750000,12063413.000000]/1000;
%radeonhd7970=[292.214996,250.309006,329.756012,489.260986,1059.537964,2959.738037,9537.487305,39747.687500]/1000;
size=32*2.^[0:7];
figure(1)
line=3;
loglog(size,gt755m,'linewidth',line,'color',c{1});
hold on;
loglog(size,apu7660,'color',c{2},'linewidth',line);
loglog(size,radeonhd5450,'color',c{3},'linewidth',line);
%loglog(size,radeonhd7970,'color',c{6},'linewidth',line);

%cpu
amd8=[946.333984,968.213989,2360.277100,8446.308594,23045.878906,117967.656250,696307.500000,2341350.750000]/1000;
apu7660=[782.661987,1626.937012,5233.705078,16983.244141,54459.621094,279185.625000,1543769.250000,4946487.500000]/1000;
%intel2600=[334.037994,633.578003,1594.543945,8484.891602,29591.263672,216617.296875,0,0]/1000;
size=32*2.^[0:7];
loglog(size,amd8,'color',c{4},'linewidth',line);
loglog(size,apu7660,'color',c{5},'linewidth',line);
%loglog(size,intel2600,'color',c{7},'linewidth',line);

%title('single');
h=xlabel('Problem Size');
set(h,'FontSize',15);
h=ylabel('time in s');
set(h,'FontSize',15);
title('single precision');
axis([32 32*2^7 0.0 12063.413])
legend('GeForce GT 755m','Radeon HD 7660D','AMD Radeon HD5450','AMD FX-8350','AMD A10-5800K','Location','northwest');
print('single2d','-depsc','-tiff')
%double
figure(2)
gt755md=[342.476000,337.440000,1186.071000,3878.008000,27473.386000,143513.373000,813040.807000]/1000;
apu7660d=[897.371000,1052.017000,2249.200000,6720.656000,25592.151000,107837.442000,438550.872000]/1000;
size=32*2.^[0:6];
loglog(size,gt755md,'linewidth',line,'color',c{1});
hold on;
loglog(size,apu7660d,'linewidth',line,'color',c{2});
%cpu
amd8d=[700.595000,905.653000,2370.888000,8831.104000,30443.549000,145436.509000, 722079.493000]/1000;
apu7660d=[717.829000,1633.183000,5164.183000,19035.183000,74297.971000,304890.835000,1419023.325000]/1000;
size=32*2.^[0:6];
loglog(size,amd8d,'linewidth',line,'color',c{4});
loglog(size,apu7660d,'linewidth',line,'color',c{5});
h=xlabel('Problem Size');
set(h,'FontSize',15);
h=ylabel('time in s');
set(h,'FontSize',15);
title('double precision');
axis([32 32*2^6 0.0 1419.023])
legend('GeForce GT 755m','Radeon HD 7660D','AMD FX-8350','AMD A10-5800K','Location','northwest');
print('double2d','-depsc','-tiff')
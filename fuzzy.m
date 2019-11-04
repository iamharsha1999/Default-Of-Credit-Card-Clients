df = readmatrix('Dataset/defaultofcreditcardclients.csv','filetype','text');
inp = df(:,1:end-1);
inp = reshape(inp,[23,30000]);
t = df(:,end);
t = reshape(t,[1,30000]);

net  = patternnet([16,8],'trainlm');
net = configure(net,inp,t);
IW = normalize(0.1 * randn(16,23));
IB = normalize(0.1 * randn(16,1));
LW1 = normalize(0.1 * randn(8,16));
B1 = normalize(0.1 * randn(8,1));
OW = normalize(0.1 * randn(1,8));
B2 = normalize(0.1 * randn(1,1));

net.IW{1} = IW;
net.LW{2,1} = LW1;
net.Lw{2,2} = OW;
net.b{1} = IB;
net.b{2} = B1;
net.b{3} = B2;
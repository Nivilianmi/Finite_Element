clear all;

input_file = fopen('FEM_conststrain_input.txt','r');
cellarray=textscan(input_file,'%s');

[Y_module_1, P_ratio_1, Y_module_2, P_ratio_2, node_num, coord, elem_num, connect, fix_num, fix_node, load_num, load_element] = read_input(cellarray);

triplot(connect,coord(:,1),coord(:,2),'g');

D_1 = [(1 - P_ratio_1) P_ratio_1 0; P_ratio_1 (1 - P_ratio_1) 0; 0 0 (1 - 2 * P_ratio_1) / 2] * Y_module_1 / ((1 + P_ratio_1) * (1 - 2 * P_ratio_1));
D_2 = [(1 - P_ratio_2) P_ratio_2 0; P_ratio_2 (1 - P_ratio_2) 0; 0 0 (1 - 2 * P_ratio_2) / 2] * Y_module_2 / ((1 + P_ratio_2) * (1 - 2 * P_ratio_2));

mix = zeros(elem_num, 1);

%globla stiffness matrix
K = zeros(2 * node_num, 2 * node_num);
for i = 1:elem_num
    node_a = connect(i, 1);
    node_b = connect(i, 2);
    node_c = connect(i, 3);

    xa = coord(node_a, 1);
    ya = coord(node_a, 2);
    xb = coord(node_b, 1);
    yb = coord(node_b, 2);
    xc = coord(node_c, 1);
    yc = coord(node_c, 2);

    x = mix(i);

    K_elem = elem_stif(xa, ya, xb, yb, xc, yc, D_1, D_2, x);

    indicator = [2 * node_a - 1, 2 * node_a, 2 * node_b - 1, 2 * node_b, 2 * node_c - 1, 2 * node_c];

    for j = 1:6
        for k = 1:6
            K(indicator(j), indicator(k)) = K(indicator(j), indicator(k)) + K_elem(j, k);
        end
    end
end

%global force vector
R = zeros(2 * node_num, 1);
for i = 1:load_num
    element = load_element(i, 1);
    node_a = load_element(i, 2);
    node_b = load_element(i, 3);
    xa = coord(node_a, 1);
    ya = coord(node_a, 2);
    xb = coord(node_b, 1);
    yb = coord(node_b, 2);
    t = load_element(i, 4);
    L = sqrt((xa - xb)^2 + (ya - yb)^2);
    R(2 * node_a - 1) = R(2 * node_a - 1) + t * L / 2;
    R(2 * node_b - 1) = R(2 * node_b - 1) + t * L / 2;
end

for i = 1:fix_num
    row = 2 * (fix_node(i, 1) - 1) + fix_node(i, 2);
    for j = 1:2 * node_num
        K(row, j) = 0;
    end
    K(row, row) = 1;
    R(row) = fix_node(i, 3);
end

U = K \ R;

% K_kron = kron(inv(K), K);

%Creaate customized NN layer
%input: K and R
%output: U
%forward implementation: U = K \ R

%plot the result
for i = 1:node_num
    x = coord(i, 1);
    y = coord(i, 2);
    coord(i, 1) = x + U(2 * (i - 1) + 1);
    coord(i, 2) = y + U(2 * (i - 1) + 2);
end

hold on;
triplot(connect,coord(:,1),coord(:,2),'r');



%element stiffness matrix
function K_elem = elem_stif(xa, ya, xb, yb, xc, yc, D_1, D_2, x)
    nax = (yb - yc)/((ya - yb) * (xc - xb) - (xa - xb) * (yc - yb));
    nay = (xc - xb)/((ya - yb) * (xc - xb) - (xa - xb) * (yc - yb));
    nbx = (yc - ya)/((yb - yc) * (xa - xc) - (xb - xc) * (ya - yc));
    nby = (xa - xc)/((yb - yc) * (xa - xc) - (xb - xc) * (ya - yc));
    ncx = (ya - yb)/((yc - ya) * (xb - xa) - (xc - xa) * (yb - ya));
    ncy = (xb - xa)/((yc - ya) * (xb - xa) - (xc - xa) * (yb - ya));
    B = [nax 0 nbx 0 ncx 0; 0 nay 0 nby 0 ncy; nay nax nby nbx ncy ncx];
    A = abs((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)) / 2;
    D = D_1 * x + D_2 * (1 - x);
    K_elem = A * B' * D * B;
end

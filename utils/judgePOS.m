function pos_max = judgePOS(p1,p2,p3)

d1 = sqrt((p1(1)-p2(1)).^2+(p1(2)-p2(2)).^2);
d2 = sqrt((p1(1)-p3(1)).^2+(p1(2)-p3(2)).^2);
d3 = sqrt((p2(1)-p3(1)).^2+(p2(2)-p3(2)).^2);
pos_max = max(max(d1,d2),d3);
end


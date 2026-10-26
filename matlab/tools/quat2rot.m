function rotation = quat2rot(q)
%QUAT2ROT Scalar-first quaternion rotation matrix.
%
% This is the polynomial, unit-quaternion rotation formula. The six
% coplanarity equations use it without adding q'*q == 1, so each quaternion
% remains a projective parameter vector.

qw = q(1);
qx = q(2);
qy = q(3);
qz = q(4);

rotation = [
    qw^2 + qx^2 - qy^2 - qz^2, 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy);
    2 * (qx * qy + qw * qz),     qw^2 - qx^2 + qy^2 - qz^2, 2 * (qy * qz - qw * qx);
    2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),   qw^2 - qx^2 - qy^2 + qz^2
];

% end
% 
% function R=quat2rot(q);
% % function R=quat2rot(q);
% % Unit quaternion q to rotation matrix R
% 
% R=[ q(1)^2+q(2)^2-q(3)^2-q(4)^2, 2*q(2)*q(3)-2*q(1)*q(4), 2*q(1)*q(3)+2*q(2)*q(4);...
%     2*q(1)*q(4)+2*q(2)*q(3), q(1)^2-q(2)^2+q(3)^2-q(4)^2, 2*q(3)*q(4)-2*q(1)*q(2);...
%     2*q(2)*q(4)-2*q(1)*q(3), 2*q(1)*q(2)+2*q(3)*q(4), q(1)^2-q(2)^2-q(3)^2+q(4)^2];

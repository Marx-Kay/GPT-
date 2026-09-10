function f=cgm_income(age,p)
b=p.income_coefficients;f=exp(b(1)+b(2)*age+b(3)*age.^2+b(4)*age.^3);
end

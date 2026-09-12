function F=cgm_income(age,p)
a=min(age,p.last_work_age);b=p.income_coefficients;
F=exp(p.income_log_level+b(1)+b(2)*a+b(3)*a.^2+b(4)*a.^3);
end

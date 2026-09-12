function write_json(path,data)
f=fopen(path,'w');assert(f>=0);clean=onCleanup(@()fclose(f));fprintf(f,'%s',jsonencode(data,PrettyPrint=true));
end

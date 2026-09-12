function record=cgm_provenance(root)
manifest=jsondecode(fileread(fullfile(root,'inputs/provenance/manifest.json')));
for i=1:numel(manifest)
 if iscell(manifest),entry=manifest{i};else,entry=manifest(i);end
 f=fullfile(root,entry.file);assert(isfile(f));
 assert(strcmp(sha(f),entry.sha256),'CGM:provenance','Registered input changed: %s',entry.file);
end
files=[dir(fullfile(root,'src','*.m'));dir(fullfile(root,'tests','*.m'));dir(fullfile(root,'run_all.m'))];
record=struct('file',{},'sha256',{});
for i=1:numel(files)
 f=fullfile(files(i).folder,files(i).name);record(i).file=erase(f,[root filesep]);record(i).sha256=sha(f);
end
write_json(fullfile(root,'results/validation/source_hashes.json'),record);
end
function hex=sha(path)
f=fopen(path,'rb');b=fread(f,Inf,'*uint8');fclose(f);
d=java.security.MessageDigest.getInstance('SHA-256');d.update(b);v=typecast(d.digest(),'uint8');hex=lower(reshape(dec2hex(v,2)',1,[]));
end

function lfs.getExt(file)
  local rev = string.reverse(file)
  local len = rev:find("%.")
  return string.reverse(rev:sub(1,len))
end

function lfs.files(dir, ext)
  local iter, dirObj = lfs.dir(dir)
  return function() -- iterator function
    local item = iter(dirObj)
    while item do
      if lfs.attributes(path.join(dir, item)).mode == "file" then -- is file
        if ext == nil or ext == lfs.getExt(item) then
          return item
        end
      end
      item = iter(dirObj)
    end
    return nil -- no more items
  end -- end of iterator
end

function lfs.mostRecent(dir)
  local mrFile, mrfMod = nil, 0
  for f in lfs.files(dir) do
    local file = path.join(dir,f)
    local mod = lfs.attributes(file).modification
    if mod > mrfMod then
      mrfMod = mod
      mrFile = f
    end
  end
  return mrFile
end

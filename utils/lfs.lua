function lfs.explode(path)
  return string.match(path, "(.-)([^\\/]-([^%.]+))$")
end

function lfs.getExt(file)
  local _, _, ext = lfs.explode(file)
  return ext
end

function lfs.getFilename(path)
  local _, fn, _ = lfs.explode(path)
  return fn
end

function lfs.getDirectory(path)
  local dir, _, _ = lfs.explode(path)
  return dir
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

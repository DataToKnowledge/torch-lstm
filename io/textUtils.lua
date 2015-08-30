-- split a string based on the delim pattern
function split(str, delim)
  local result,pat,lastPos = {},"(.-)" .. delim .. "()",1
  for part, pos in string.gfind(str, pat) do
    if part and part ~= '' then
      table.insert(result, part); lastPos = pos
    end
  end
  local lastPart = string.sub(str, lastPos)
  if lastPart and lastPart ~= '' then
    table.insert(result, lastPart)
  end
  return result
end

function splitByLine(str)
  return split(str, '\r?\n\n?') -- the last \n? match last line with double \n\n
end

function splitBySpace(str)
  return split(str,' ')
end

local M = { }

function script_path()
   local str = debug.getinfo(2, "S").source:sub(2)
   return str:match("(.*/)")
end

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('OpenFace')
   cmd:text()
   cmd:text('Options:')

   ------------ General options --------------------
   cmd:option('-outDir', './reps/', 'Subdirectory to output the representations')
   cmd:option('-data','Home of dataset')
   cmd:option('-model','nn4.small2.v1.t7')
   cmd:option('-imgDim', 96, 'Image dimension. nn1=224, nn4=96')
   cmd:option('-batchSize',       50,   'mini-batch size')
   cmd:option('-cuda',       false,   'Use cuda')
   cmd:option('-device',       1,   'Cuda device to use')
   cmd:option('-cache',       false,   'Cache loaded data.')
   cmd:text()

   local opt = cmd:parse(arg or {})
   os.execute('mkdir -p ' .. opt.outDir)

   return opt
end

return M

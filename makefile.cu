.cu.o:
	@NVCC_PATH@ -dc $(AM_CPPFLAGS) --maxrregcount 32 @NVCC_CPPFLAGS@ @NVCC_ARCHITECTURE@ $< -o $@
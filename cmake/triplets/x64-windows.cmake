# Overlay of vcpkg's built-in x64-windows triplet (same name, so it takes precedence when
# VCPKG_OVERLAY_TRIPLETS points to this directory).
#
# HDF5 and its compression dependencies are linked statically into PBAT. Other Python packages
# (i.e. h5py) bundle their own, differently versioned, but identically named hdf5.dll/z.dll.
# Windows loads a single DLL per name per process, so whichever package is imported first would
# otherwise force its HDF5 onto the other.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_PROVIDED_FORTRAN ON)

if(PORT MATCHES "^(hdf5|zlib|libaec)$")
    set(VCPKG_LIBRARY_LINKAGE static)
endif()

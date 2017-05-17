use libc::{c_void, c_char, c_int, c_float, size_t};

pub struct cudnnContext {}
pub type cudnnHandle_t = *mut cudnnContext;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum cudnnStatus_t {
  Success         = 0,
  NotInitialized  = 1,
  AllocFailed     = 2,
  BadParam        = 3,
  InternalError   = 4,
  InvalidError    = 5,
  ArchMismatch    = 6,
  MappingError    = 7,
  ExecutionFailed = 8,
  NotSupported    = 9,
  LicenseError    = 10,
  RuntimePrerequisiteMissing = 11
}

impl Default for cudnnStatus_t {
  fn default() -> cudnnStatus_t {
    cudnnStatus_t::Success
  }
}

impl cudnnStatus_t {
  pub fn is_ok(&self) -> bool {
    !self.is_err()
  }

  pub fn is_err(&self) -> bool {
    if let cudnnStatus_t::Success = *self {
      false
    } else {
      true
    }
  }
}

pub enum cudnnTensorStruct {}
pub type  *mut cudn: *cudnnTensorDescriptor_tnTensorStruct;

pub enum cudnnConvolutionStruct {}
pub type cudnnConvolutionDescriptor_t = *mut cudnnConvolutionStruct;

pub enum cudnnPoolingStruct {}
pub type cudnnPoolingDescriptor_t = *mut cudnnPoolingStruct;

pub enum cudnnFilterStruct {}
pub type cudnnFilterDescriptor_t = *mut cudnnFilterStruct;

pub enum cudnnLRNStruct {}
pub type cudnnLRNDescriptor_t = *mut cudnnLRNStruct;

pub enum cudnnActivationStruct {}
pub type cudnnActivationDescriptor_t = *mut cudnnActivationStruct;

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnDataType_t {
    Float   = 0,
    Double  = 1,
    Half    = 2,
    int8    = 3,
    int32   = 4,
    int8x4  = 5
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnTensorFormat_t {
    NCHW  = 0,
    NHWC  = 1,
    NCHW_VECT_C = 2
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnAddMode_t {
    //SameHW      = 0,
    Image       = 0,
    //SameCHW     = 1,
    FeatureMap  = 1,
    SameC       = 2,
    FullTensor  = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnConvolutionMode_t {
    Convolution       = 0,
    CrossCorrelation  = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnConvolutionFwdPreference_t {
    NoWorkspace           = 0,
    PreferFastest         = 1,
    SpecifyWorkspaceLimit = 2,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum cudnnConvolutionFwdAlgo_t {
    ImplicitGemm        = 0,
    ImplicitPrecompGemm = 1,
    Gemm                = 2,
    Direct              = 3,
    Fft                 = 4,
    FftTiling           = 5,
    Winograd            = 6,
    WinogradNonfused    = 7,
}

impl Default for cudnnConvolutionFwdAlgo_t {
    fn default() -> cudnnConvolutionFwdAlgo_t {
        cudnnConvolutionFwdAlgo_t::ImplicitGemm
    }
}

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct cudnnConvolutionFwdAlgoPerf_t {
    pub algo:   cudnnConvolutionFwdAlgo_t,
    pub status: cudnnStatus_t,
    pub time:   c_float,
    pub memory: size_t,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnConvolutionBwdFilterPreference_t {
    NoWorkspace           = 0,
    PreferFastest         = 1,
    SpecifyWorkspaceLimit = 2,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum cudnnConvolutionBwdFilterAlgo_t {
    NonDeterministic          = 0,
    Deterministic             = 1,
    Fft                       = 2,
    NonDeterministicWorkspace = 3,
    WinogradNonfused          = 4,
}

impl Default for cudnnConvolutionBwdFilterAlgo_t {
    fn default() -> cudnnConvolutionBwdFilterAlgo_t {
        cudnnConvolutionBwdFilterAlgo_t::NonDeterministic
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct cudnnConvolutionBwdFilterAlgoPerf_t {
    pub algo:   cudnnConvolutionBwdFilterAlgo_t,
    pub status: cudnnStatus_t,
    pub time:   c_float,
    pub memory: size_t,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnConvolutionBwdDataPreference_t {
    NoWorkspace           = 0,
    PreferFastest         = 1,
    SpecifyWorkspaceLimit = 2,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum cudnnConvolutionBwdDataAlgo_t {
    NonDeterministic  = 0,
    Deterministic     = 1,
    Fft               = 2,
    FftTiling         = 3,
    Winograd          = 4,
    WinogradNonfused  = 5,
}

impl Default for cudnnConvolutionBwdDataAlgo_t {
    fn default() -> cudnnConvolutionBwdDataAlgo_t {
        cudnnConvolutionBwdDataAlgo_t::NonDeterministic
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct cudnnConvolutionBwdDataAlgoPerf_t {
    pub algo:   cudnnConvolutionBwdDataAlgo_t,
    pub status: cudnnStatus_t,
    pub time:   c_float,
    pub memory: size_t,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnSoftmaxAlgorithm_t {
    Fast      = 0,
    Accurate  = 1,
    Log       = 2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnSoftmaxMode_t {
    Instance  = 0,
    Channel   = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnPoolingMode_t {
    Max                           = 0,
    AverageCountIncludingPadding  = 1,
    AverageCountExcludingPadding  = 2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnActivationMode_t {
    Sigmoid     = 0,
    Relu        = 1,
    Tanh        = 2,
    ClippedRelu = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnNanPropagation_t {
    NotPropagateNan = 0,
    PropagateNan    = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnBatchNormMode_t {
    PerActivation = 0,
    Spatial       = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnDeterminism_t {
    NoneDeterministic = 0,
    Deterministic     = 1,
};

#[derive(Clone, Copy)]
#[repr(C)]
enum cudnnOpTensorOp_t {
    Add  = 0,
    Mul  = 1,
    Min  = 2,
    Max  = 3,
    Sqrt = 4,
};

#[derive(Clone, Copy)]
#[repr(C)]
enum cudnnReduceTensorOp_t {
    Add   = 0,
    Mul   = 1,
    Min   = 2,
    Max   = 3,
    AMax  = 4,
    Avg   = 5,
    Norm1 = 6,
    Norm2 = 7,
};

#[derive(Clone, Copy)]
#[repr(C)]
enum cudnnReduceTensorIndices_t {
    NoIndicies       = 0,
    FlattenedIndices = 1,
};

#[derive(Clone, Copy)]
#[repr(C)]
enum cudnnIndicesType_t {
    u32Indices = 0,
    u64Indices = 1,
    u16Indices = 2,
    u8Indices  = 3,
};

impl Default for cudnnIndicesType_t {
    fn default() -> cudnnIndicesType_t {
        cudnnIndicesType_t::u32Indices
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
enum cudnnConvolutionMode_t {
    Convolution       = 0,
    CrossCorrelation  = 1
};

#[link(name = "cudnn", kind = "dylib")]
extern "C" {
    pub fn cudnnGetVersion() -> size_t;
    pub fn cudnnGetCudartVersion() -> size_t;
    pub fn cudnnGetErrorString(status: cudnnStatus_t) -> *const c_char;

    pub fn cudnnCreate (handle: *cudnnHandle_t) -> cudnnStatus_t;
    pub fn cudnnDestroy (handle: cudnnHandle_t) -> cudnnStatus_t;
    pub fn cudnnSetStream (handle: cudnnHandle_t, streamId: cudaStream_t) -> cudnnStatus_t;
    pub fn cudnnGetStream (handle: cudnnHandle_t, streamId: *cudaStream_t) -> cudnnStatus_t;

    pub fn cudnnCreateTensorDescriptor(tensorDesc: *cudnnTensorDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t;

    pub fn cudnnSetTensor4dDescriptor(tensorDesc: cudnnTensorDescriptor_t,
                                      format: cudnnTensorFormat_t,
                                      dataType: cudnnDataType_t, // image data type
                                      n: c_int,        // number of inputs (batch size)
                                      c: c_int,        // number of input feature maps
                                      h: c_int,        // height of input section
                                      w: c_int)       // width of input section
                                      -> cudnnStatus_t;
    pub fn cudnnSetTensor4dDescriptorEx(tensorDesc: cudnnTensorDescriptor_t,
                                        dataType: cudnnDataType_t, // image data type
                                        n: c_int, // number of inputs (batch size)
                                        c: c_int, // number of input feature maps
                                        h: c_int, // height of input section
                                        w: c_int, // width of input section
                                        nStride: c_int,
                                        cStride: c_int,
                                        hStride: c_int,
                                        wStride: c_int)
                                        -> cudnnStatus_t;
    pub fn cudnnGetTensor4dDescriptor(tensorDesc: const cudnnTensorDescriptor_t,
                                      dataType: *cudnnDataType_t, // image data type
                                      n: *c_int,        // number of inputs (batch size)
                                      c: *c_int,        // number of input feature maps
                                      h: *c_int,        // height of input section
                                      w: *c_int,        // width of input section
                                      nStride: *c_int,
                                      cStride: *c_int,
                                      hStride: *c_int,
                                      wStride: *c_int )
                                      -> cudnnStatus_t;

    pub fn cudnnSetTensorNdDescriptor(tensorDesc: cudnnTensorDescriptor_t,
                                      dataType: cudnnDataType_t,
                                      nbDims: c_int,
                                      dimA: const [c_int],
                                      strideA: const [c_int] )
                                      -> cudnnStatus_t;
    pub fn cudnnSetTensorNdDescriptorEx(tensorDesc: cudnnTensorDescriptor_t,
                                        format: cudnnTensorFormat_t,
                                        dataType: cudnnDataType_t,
                                        nbDims: c_int,
                                        dimA: const [c_int])
                                        -> cudnnStatus_t;
    pub fn cudnnGetTensorNdDescriptor(tensorDesc: const cudnnTensorDescriptor_t,
                                      nbDimsRequested: c_int,
                                      dataType: *cudnnDataType_t,
                                      nbDims: *c_int,
                                      dimA: [c_int],
                                      strideA: [c_int] )
                                      -> cudnnStatus_t;

    pub fn cudnnGetTensorSizeInBytes(tensorDesc: const cudnnTensorDescriptor_t,
                                    size: *size_t)
                                    -> cudnnStatus_t;

    pub fn cudnnTransformTensor(handle: cudnnHandle_t,
                                alpha: const *c_void,
                                xDesc: const cudnnTensorDescriptor_t,
                                x: const *c_void,
                                beta: const *c_void,
                                yDesc: const cudnnTensorDescriptor_t,
                                y: *c_void)
                                -> cudnnStatus_t;

    pub fn cudnnAddTensor(handle: cudnnHandle_t,
                          alpha: const *c_void,
                          aDesc: const cudnnTensorDescriptor_t,
                          A: const *c_void,
                          beta: const *c_void,
                          cDesc: const cudnnTensorDescriptor_t,
                          C: *c_void )
                          -> cudnnStatus_t;
    
    /** OP TENSOR **/
    pub fn cudnnCreateOpTensorDescriptor(opTensorDesc: *cudnnOpTensorDescriptor_t ) -> cudnnStatus_t;
    pub fn cudnnDestroyOpTensorDescriptor(opTensorDesc: cudnnOpTensorDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetOpTensorDescriptor(opTensorDesc: cudnnOpTensorDescriptor_t,
                                      opTensorOp: cudnnOpTensorOp_t,
                                      opTensorCompType: cudnnDataType_t,
                                      opTensorNanOpt: cudnnNanPropagation_t )
                                      -> cudnnStatus_t;
    pub fn cudnnGetOpTensorDescriptor(opTensorDesc: const cudnnOpTensorDescriptor_t,
                                      opTensorOp: *cudnnOpTensorOp_t,
                                      opTensorCompType: *cudnnDataType_t,
                                      opTensorNanOpt: *cudnnNanPropagation_t)
                                      -> cudnnStatus_t;
    
    /** REDUCE TENSOR **/
    pub fn cudnnCreateReduceTensorDescriptor(reduceTensorDesc: *cudnnReduceTensorDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetReduceTensorDescriptor(reduceTensorDesc: cudnnReduceTensorDescriptor_t,
                                          reduceTensorOp: cudnnReduceTensorOp_t,
                                          reduceTensorCompType: cudnnDataType_t,
                                          reduceTensorNanOpt: cudnnNanPropagation_t,
                                          reduceTensorIndices: cudnnReduceTensorIndices_t,
                                          reduceTensorIndicesType: cudnnIndicesType_t )
                                          -> cudnnStatus_t;
    pub fn cudnnGetReduceTensorDescriptor(reduceTensorDesc: const cudnnReduceTensorDescriptor_t,
                                          reduceTensorOp: *cudnnReduceTensorOp_t,
                                          reduceTensorCompType: *cudnnDataType_t,
                                          reduceTensorNanOpt: *cudnnNanPropagation_t,
                                          reduceTensorIndices: *cudnnReduceTensorIndices_t,
                                          reduceTensorIndicesType: *cudnnIndicesType_t)
                                          -> cudnnStatus_t;
    pub fn cudnnDestroyReduceTensorDescriptor(reduceTensorDesc: cudnnReduceTensorDescriptor_t) -> cudnnStatus_t;

    /* Helper function to return the minimum size of the index space to be passed to the reduction given the input and output tensors */
    pub fn cudnnGetReductionIndicesSize(handle: cudnnHandle_t,
                                        reduceTensorDesc: const cudnnReduceTensorDescriptor_t,
                                        aDesc: const cudnnTensorDescriptor_t,
                                        cDesc: const cudnnTensorDescriptor_t,
                                        sizeInBytes: *size_t )
                                        -> cudnnStatus_t;

    /* Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors */
    pub fn cudnnGetReductionWorkspaceSize(handle: cudnnHandle_t,
                                          reduceTensorDesc: const cudnnReduceTensorDescriptor_t,
                                          aDesc: const cudnnTensorDescriptor_t,
                                          cDesc: const cudnnTensorDescriptor_t,
                                          sizeInBytes: *size_t )
                                          -> cudnnStatus_t;
    /* Tensor operation : C = reduce op( alpha * A ) + beta * C */
    /* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
    /* The indices space is ignored for reduce ops other than min or max. */
    pub fn cudnnReduceTensor(handle: cudnnHandle_t,
                             reduceTensorDesc: const cudnnReduceTensorDescriptor_t,
                             indices: *c_void,
                             indicesSizeInBytes: size_t,
                             workspace: *c_void,
                             workspaceSizeInBytes: size_t,
                             alpha: const *c_void,
                             aDesc: const cudnnTensorDescriptor_t,
                             A: const *c_void,
                             beta: const *c_void,
                             cDesc: const cudnnTensorDescriptor_t,
                             C: *c_void )
                             -> cudnnStatus_t;
    /* Set all values of a tensor to a given value : y[i] = value[0] */
    pub fn cudnnSetTensor(handle: cudnnHandle_t,
                          yDesc: const cudnnTensorDescriptor_t,
                          y: *c_void,
                          valuePtr: const *void)
                          -> cudnnStatus_t;
    /* Scale all values of a tensor by a given factor : y[i] = alpha * y[i] */
    pub fn cudnnScaleTensor(handle: cudnnHandle_t,
                            yDesc: const cudnnTensorDescriptor_t,
                            y: *c_void,
                            alpha: const *c_void )
                            -> cudnnStatus_t;
}
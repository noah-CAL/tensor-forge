//! Representations for dense, multidimensional arrays stored in contiguous memory.

/// Error types for Tensor construction.
#[derive(Debug)]
pub enum TensorError {
    /// Raised in [`Tensor::from_vec`] or [`Tensor::zeros`] if 
    /// one of a tensor's dimensions is zero.
    InvalidShape,
    /// Raised if the tensor shape does not match the 
    /// shape of the data passed in to [`Tensor::from_vec`].
    ShapeMismatch
}

/// Representation for a multidimensional array of numbers.
///
/// Tensors are used as data inputs to ML operations and are 
/// the basic datatype for the `tensor-forge` library.
///
/// # Examples
/// ```
/// # use tensor_forge::tensor::Tensor;
/// # // TODO: add examples of later use with Graphs
/// # // and flesh out documentation
/// let shape = vec![4, 4];
/// let data: Vec<f64> = (0..16).map(|x| x as f64).collect();
/// let tensor = Tensor::from_vec(shape, data);
/// assert!(tensor.is_ok());
/// ```
///
/// Data will be stored in a contiguous array of IEEE 754 double-precision floating-point. 
#[derive(Debug)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

impl Tensor {
    /// Constructs a zero-filled [`Tensor`] of a given shape.
    ///
    /// Use [`Tensor::from_vec`] to construct a tensor with
    /// data, or fill the tensor after a call to [`Tensor::data_mut`]. 
    ///
    /// # Errors
    /// - [`TensorError::InvalidShape`] if shape contains a zeroed dimension.
    /// - [`TensorError::ShapeMismatch`] if shape contains a zeroed dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensor_forge::tensor::Tensor;
    /// let shape = vec![4, 4];
    /// let tensor = Tensor::zeros(shape);
    /// assert_eq!(tensor.unwrap().data(), [0_f64; 4 * 4]);
    /// ```
    ///
    /// Data will be stored in a contiguous array of IEEE 754 double-precision floating-point. 
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Tensor, TensorError> { 
        let shape: Vec<usize> = shape.into();
        let num_elements = shape.iter().product();
        let mut data = Vec::with_capacity(num_elements);
        data.resize(num_elements, 0_f64);
        Tensor::from_vec(shape, data)
    }

    /// Constructs a [`Tensor`] of the given dimensions in `shape` from the input `data`.
    ///
    /// # Errors
    /// - [`TensorError::InvalidShape`] if shape contains a zeroed dimension.
    /// - [`TensorError::ShapeMismatch`] if the data cannot fit into the tensor's dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensor_forge::tensor::Tensor;
    /// // Example (toy) data
    /// let data: Vec<f64> = (0..16).map(|x| x as f64).collect();
    /// let shape = vec![4, 4];
    ///
    /// let tensor = Tensor::from_vec(shape, data);
    /// assert!(tensor.is_ok());
    /// ```
    ///
    /// Tensor data can fit into multiple valid shapes. For example, the above data with 16 total elements can fit into 
    /// an 8x2, 2x8, 1x16, or 16x1 tensor.
    ///
    /// ```
    /// # use tensor_forge::tensor::Tensor;
    /// // Example (toy) data from above
    /// let data: Vec<f64> = (0..16).map(|x| x as f64).collect();
    /// let shape = vec![16, 1];
    ///
    /// let tensor = Tensor::from_vec(shape, data);
    /// assert!(tensor.is_ok());
    /// ```
    pub fn from_vec(shape: impl Into<Vec<usize>>, data: Vec<f64>) -> Result<Tensor, TensorError> {
        let shape: Vec<usize> = shape.into();
        let num_elements: usize = shape.iter().product();
        if num_elements == 0 {
            return Err(TensorError::InvalidShape);
        }
        if num_elements != data.len() {
            return Err(TensorError::ShapeMismatch);
        }
        Ok(Tensor {
            shape,
            data,
        })
    }

    /// Returns the shape of this tensor.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::tensor::Tensor;
    /// let shape = vec![4, 4];
    /// let tensor = Tensor::zeros(shape);
    /// assert_eq!(tensor.unwrap().shape(), vec![4, 4]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the total number of elements in this tensor.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::tensor::Tensor;
    /// let shape = vec![4, 4];
    /// let tensor = Tensor::zeros(shape);
    /// assert_eq!(tensor.unwrap().numel(), 16);  // 4x4 = 16 elements
    /// ```
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns an immutable reference to the data in this tensor.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::tensor::Tensor;
    /// let shape = vec![4, 4];
    /// let tensor = Tensor::zeros(shape);
    /// assert_eq!(tensor.unwrap().data(), vec![0_f64; 4 * 4]);
    /// ```
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Returns a mutable reference to the data in this tensor.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::tensor::Tensor;
    /// let shape = vec![4, 4];
    /// let mut tensor = Tensor::zeros(shape);
    /// assert_eq!(tensor.unwrap().data_mut(), vec![0_f64; 4 * 4]);
    /// ```
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }
}

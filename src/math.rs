use std::{
    fmt::Display,
    os::raw::{c_int, c_ulong, c_void},
};

#[allow(non_camel_case_types)]
type cudaError_t = c_int;

#[link(name = "cudart")]
unsafe extern "C" {
    unsafe fn cudaMalloc(ptr: *mut *mut c_void, size: c_ulong) -> cudaError_t;
    unsafe fn cudaFree(ptr: *mut c_void) -> cudaError_t;
    unsafe fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: c_ulong,
        kind: c_int,
    ) -> cudaError_t;
}

#[link(name = "cuda_lib", kind = "static")]
unsafe extern "C" {
    unsafe fn t(a_ptr: *mut c_void, d_ptr: *mut c_void, r: c_int, c: c_int);
    unsafe fn mmul(a_ptr: *mut c_void, b_ptr: *mut c_void, r: c_int, c: c_int) -> *mut c_void;
    unsafe fn emul(a_ptr: *mut c_void, b_ptr: *mut c_void, d_ptr: *mut c_void, n: c_int);
    unsafe fn eadd(a_ptr: *mut c_void, b_ptr: *mut c_void, d_ptr: *mut c_void, n: c_int);
    unsafe fn esub(a_ptr: *mut c_void, b_ptr: *mut c_void, d_ptr: *mut c_void, n: c_int);
    unsafe fn unary_op(a_ptr: *mut c_void, d_ptr: *mut c_void, op: c_int, n: c_int);
}

// cudaMemcpyKind enum
pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
pub enum UnaryOp {
    Relu = 0,
    LeakyRelu = 1,
    Silu = 2,
    Gelu = 3,
    Tanh = 4,
    Sigmoid = 5,
    Softplus = 6,
}

pub trait MatFamily {
    type Mat<const R: usize, const C: usize>: Matrix<R, C, Fam = Self>;
}

pub trait Matrix<const R: usize, const C: usize>: Sized + Display {
    type Fam: MatFamily;

    fn from_slice(slice: &[f32]) -> anyhow::Result<Self>;
    fn zeros() -> Self;
    fn rows() -> usize;
    fn cols() -> usize;
    fn t(&self) -> <Self::Fam as MatFamily>::Mat<C, R>;
    fn mmul<const N: usize>(
        &self,
        rhs: &<Self::Fam as MatFamily>::Mat<C, N>,
    ) -> <Self::Fam as MatFamily>::Mat<R, N>;
    fn emul(&self, rhs: Self) -> Self;
    fn emul_assign(&mut self, rhs: Self);
    fn add(&self, rhs: Self) -> Self;
    fn add_assign(&mut self, rhs: Self);
    fn sub(&self, rhs: Self) -> Self;
    fn sub_assign(&mut self, rhs: Self);
    fn unary_op(&self, op: UnaryOp) -> Self;
    fn unary_op_assign(&mut self, op: UnaryOp);
}

pub struct HostFam {}

pub struct HostMatrix<const R: usize, const C: usize> {
    data: Vec<f32>,
}

impl<const R: usize, const C: usize> HostMatrix<R, C> {
    fn get(&self, r: usize, c: usize) -> Option<&f32> {
        self.data.get(r * C + c)
    }

    fn set(&mut self, val: f32, r: usize, c: usize) {
        if let Some(value) = self.data.get_mut(r * C + c) {
            *value = val;
        }
    }
}

impl MatFamily for HostFam {
    type Mat<const R: usize, const C: usize> = HostMatrix<R, C>;
}

impl<const R: usize, const C: usize> Matrix<R, C> for HostMatrix<R, C> {
    type Fam = HostFam;

    fn from_slice(value: &[f32]) -> anyhow::Result<Self> {
        if value.len() != R * C {
            Err(anyhow::Error::msg("Supplied slice was of incorrect length"))
        } else {
            Ok(Self {
                data: Vec::from(value),
            })
        }
    }

    fn zeros() -> Self {
        Self {
            data: vec![0.0_f32; R * C],
        }
    }

    fn rows() -> usize {
        R
    }

    fn cols() -> usize {
        C
    }

    fn mmul<const N: usize>(
        &self,
        rhs: &<Self::Fam as MatFamily>::Mat<C, N>,
    ) -> <Self::Fam as MatFamily>::Mat<R, N> {
        let mut result: HostMatrix<R, N> = HostMatrix::zeros();

        for r in 0..R {
            for c in 0..N {
                let mut val: f32 = 0.0;

                for n in 0..C {
                    val += self.get(r, n).unwrap() * rhs.get(n, c).unwrap();
                }

                result.set(val, r, c);
            }
        }

        result
    }

    fn emul(&self, rhs: Self) -> Self {
        let result_data: Vec<f32> = self.data.iter().zip(rhs.data).map(|(a, b)| a * b).collect();
        Self::from(result_data)
    }

    fn emul_assign(&mut self, rhs: Self) {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i] * rhs.data[i]
        }
    }

    fn add(&self, rhs: Self) -> Self {
        let result_data: Vec<f32> = self.data.iter().zip(rhs.data).map(|(a, b)| a + b).collect();
        Self::from(result_data)
    }

    fn add_assign(&mut self, rhs: Self) {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i] + rhs.data[i]
        }
    }

    fn sub(&self, rhs: Self) -> Self {
        let result_data: Vec<f32> = self.data.iter().zip(rhs.data).map(|(a, b)| a - b).collect();
        Self::from(result_data)
    }

    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i] - rhs.data[i]
        }
    }

    fn unary_op(&self, op: UnaryOp) -> Self {
        let unary = match op {
            UnaryOp::Relu => |x: &f32| if *x < 0.0 { 0.0 } else { *x },
            UnaryOp::LeakyRelu => |x: &f32| if *x < 0.0 { 0.01 * *x } else { *x },
            UnaryOp::Silu => |x: &f32| {
                // *x * sigmoid(*x)
                let e = (-*x).exp();
                *x / (1.0 + e)
            },
            UnaryOp::Gelu => |x: &f32| {
                // Appro*ximate GELU: 0.5**x*(1+tanh(√(2/π)*(*x+0.044715*x³)))
                let u = 0.7978845608 * (*x + 0.044715 * *x * *x * *x);
                0.5 * *x * (1.0 + u.tanh())
            },
            UnaryOp::Tanh => |x: &f32| (*x).tanh(),
            UnaryOp::Sigmoid => |x: &f32| {
                if *x >= 0.0 {
                    1.0 / (1.0 + (-*x).exp())
                } else {
                    let e = (*x).exp();
                    e / (1.0 + e)
                }
            },
            UnaryOp::Softplus => |x: &f32| {
                // ln(1+e^*x) with overflow protection
                if *x > 20.0 {
                    *x // large positive: log(1+e^*x) ≈ *x
                } else if *x < -20.0 {
                    (-*x).exp() // very negative: log(1+tiny) ≈ tiny
                } else {
                    (1.0 + (*x).exp()).ln()
                }
            },
        };

        Self::from_slice(&self.data.iter().map(unary).collect::<Vec<f32>>()).unwrap()
    }

    fn unary_op_assign(&mut self, op: UnaryOp) {
        let unary = match op {
            UnaryOp::Relu => |x: &f32| if *x < 0.0 { 0.0 } else { *x },
            UnaryOp::LeakyRelu => |x: &f32| if *x < 0.0 { 0.01 * *x } else { *x },
            UnaryOp::Silu => |x: &f32| {
                // *x * sigmoid(*x)
                let e = (-*x).exp();
                *x / (1.0 + e)
            },
            UnaryOp::Gelu => |x: &f32| {
                // Appro*ximate GELU: 0.5**x*(1+tanh(√(2/π)*(*x+0.044715*x³)))
                let u = 0.7978845608 * (*x + 0.044715 * *x * *x * *x);
                0.5 * *x * (1.0 + u.tanh())
            },
            UnaryOp::Tanh => |x: &f32| (*x).tanh(),
            UnaryOp::Sigmoid => |x: &f32| {
                if *x >= 0.0 {
                    1.0 / (1.0 + (-*x).exp())
                } else {
                    let e = (*x).exp();
                    e / (1.0 + e)
                }
            },
            UnaryOp::Softplus => |x: &f32| {
                // ln(1+e^*x) with overflow protection
                if *x > 20.0 {
                    *x // large positive: log(1+e^*x) ≈ *x
                } else if *x < -20.0 {
                    (-*x).exp() // very negative: log(1+tiny) ≈ tiny
                } else {
                    (1.0 + (*x).exp()).ln()
                }
            },
        };

        self.data = self.data.iter().map(unary).collect::<Vec<f32>>()
    }

    fn t(&self) -> HostMatrix<C, R> {
        let mut result: HostMatrix<C, R> = HostMatrix::zeros();

        for r in 0..R {
            for c in 0..C {
                result.set(*self.get(r, c).unwrap(), c, r)
            }
        }

        result
    }
}

impl<const R: usize, const C: usize> From<CudaMatrix<R, C>> for anyhow::Result<HostMatrix<R, C>> {
    fn from(value: CudaMatrix<R, C>) -> Self {
        let mut data = vec![0.0f32; R * C];
        let bytes = R * C * std::mem::size_of::<f32>();
        unsafe {
            let err = cudaMemcpy(
                data.as_mut_ptr() as *mut _,
                value.ptr as *const _,
                bytes as _,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            if err != 0 {
                panic!("cudaMemcpy D2H failed with error code {}", err);
            }
        }
        Ok(HostMatrix::<R, C> { data })
    }
}

impl<const R: usize, const C: usize> From<Vec<f32>> for HostMatrix<R, C> {
    fn from(value: Vec<f32>) -> Self {
        Self { data: value }
    }
}

fn right_pad(string: &String, length: usize, pad_char: char) -> String {
    if string.len() >= length {
        string.clone()
    } else {
        let mut result: String = String::with_capacity(length);
        result.push_str(string);
        result.push_str(&vec![pad_char.to_string(); length - string.len()].join(""));
        result
    }
}

impl<const R: usize, const C: usize> Display for HostMatrix<R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut rows: Vec<Vec<String>> = Vec::with_capacity(R);
        for r in 0..R {
            let mut row: Vec<String> = Vec::with_capacity(C);
            for c in 0..C {
                row.push(self.get(r, c).unwrap().to_string());
            }

            rows.push(row);
        }

        for c in 0..C {
            let mut max_len: usize = 0;

            for r in 0..R {
                if rows[r][c].len() > max_len {
                    max_len = rows[r][c].len()
                }
            }

            for r in 0..R {
                rows[r][c] = right_pad(&rows[r][c], max_len, ' ');
            }
        }

        write!(
            f,
            "{}",
            rows.iter()
                .map(|row| format!("| {} |", row.join(" ")))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }
}

pub struct CudaFam {}

pub struct CudaMatrix<const R: usize, const C: usize> {
    pub ptr: *mut f32, // device memory
}

impl MatFamily for CudaFam {
    type Mat<const R: usize, const C: usize> = CudaMatrix<R, C>;
}

impl<const R: usize, const C: usize> Matrix<R, C> for CudaMatrix<R, C> {
    type Fam = CudaFam;

    fn from_slice(slice: &[f32]) -> anyhow::Result<Self> {
        Ok(Self::from(HostMatrix::<R, C>::from_slice(slice)?))
    }
    fn zeros() -> Self {
        todo!()
    }

    fn rows() -> usize {
        todo!()
    }

    fn cols() -> usize {
        todo!()
    }

    fn mmul<const N: usize>(
        &self,
        rhs: &<Self::Fam as MatFamily>::Mat<C, N>,
    ) -> <Self::Fam as MatFamily>::Mat<R, N> {
        todo!()
    }

    fn emul(&self, rhs: Self) -> Self {
        todo!()
    }

    fn emul_assign(&mut self, rhs: Self) {
        todo!()
    }

    fn add(&self, rhs: Self) -> Self {
        todo!()
    }

    fn add_assign(&mut self, rhs: Self) {
        todo!()
    }

    fn sub(&self, rhs: Self) -> Self {
        todo!()
    }

    fn sub_assign(&mut self, rhs: Self) {
        todo!()
    }

    fn unary_op(&self, op: UnaryOp) -> Self {
        todo!()
    }

    fn unary_op_assign(&mut self, op: UnaryOp) {
        todo!()
    }

    fn t(&self) -> CudaMatrix<C, R> {
        todo!()
    }
}

impl<const R: usize, const C: usize> From<HostMatrix<R, C>> for CudaMatrix<R, C> {
    fn from(value: HostMatrix<R, C>) -> Self {
        todo!()
    }
}

impl<const R: usize, const C: usize> Drop for CudaMatrix<R, C> {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                let _ = cudaFree(self.ptr as *mut _);
                self.ptr = std::ptr::null_mut();
            }
        }
    }
}

impl<const R: usize, const C: usize> Display for CudaMatrix<R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

import numpy as np
from pprint import PrettyPrinter

class BlockDiagonalMatrix:
    def __init__(self, blocks: np.ndarray) -> None:
        """
        Initialize the Block Diagonal Matrix.
        
        :param blocks: 2D NumPy array where each element is a NumPy array representing 
                       the diagonal elements of a diagonal block.
        """
        self.blocks = blocks
        self.n = self.blocks.shape[0]  # Number of blocks along one dimension
        self.d = self.blocks.shape[2]  # Size of each block (assuming non-empty blocks)
    
    def numpy(self) -> np.ndarray:
        """
        Convert the block diagonal matrix into a NumPy array representation.
        This method constructs the full matrix from the block diagonal representation and 
        returns it as a NumPy array. 

        :return: A NumPy array representing the full sparse matrix.
        """
        # Initialize a full matrix of zeros with the appropriate size
        full_matrix = np.zeros((self.n * self.d, self.n * self.d))

        # Fill in the diagonal blocks
        for i in range(self.n):
            for j in range(self.n):
                if np.any(self.blocks[i, j] != 0):  # Only process non-zero blocks
                    np.fill_diagonal(
                        full_matrix[i * self.d:(i + 1) * self.d, j * self.d:(j + 1) * self.d],
                        self.blocks[i, j]
                    )

        return full_matrix
    
    def __str__(self) -> str:
        """
        Constructs the full matrix from the block diagonal representation and
        returns its string representation. It fills in the diagonal blocks and converts the 
        full matrix into a string format.

        :return: A string representation of the full matrix.
        """
        return str(self.numpy())
    
    def _repr_pretty_(self, p: "PrettyPrinter", cycle: bool) -> None:
        """
        Provides a more readable representation of the matrix when used in Jupyter
        notebooks. It helps to visualize the block diagonal matrix in a user-friendly format.

        :param p: The PrettyPrinter object.
        :param cycle: Flag to indicate if there is a cycle in the object graph.
        """
        p.text(str(self) if not cycle else "...")


    def __add__(self, other: "BlockDiagonalMatrix") -> "BlockDiagonalMatrix":
        """        
        This method adds two block diagonal matrices element-wise. The result is a new
        BlockDiagonalMatrix where each block is the sum of the corresponding blocks in 
        the input matrices. It overloads the `+` operator.

        :param other: Another BlockDiagonalMatrix to add.
        :return: The sum as a new BlockDiagonalMatrix.
        """
        assert self.blocks.shape == other.blocks.shape, "Matrices must have the same dimensions."
        result_blocks = self.blocks + other.blocks
        return BlockDiagonalMatrix(result_blocks)

    def __sub__(self, other: "BlockDiagonalMatrix") -> "BlockDiagonalMatrix":
        """        
        This method subtracts two block diagonal matrices element-wise. The result is a new
        BlockDiagonalMatrix where each block is the sum of the corresponding blocks in 
        the input matrices. It overloads the `+` operator.

        :param other: Another BlockDiagonalMatrix to add.
        :return: The subtraction result as a new BlockDiagonalMatrix.
        """
        assert self.blocks.shape == other.blocks.shape, "Matrices must have the same dimensions."
        result_blocks = self.blocks - other.blocks
        return BlockDiagonalMatrix(result_blocks)
    
    def __mul__(self, other: "BlockDiagonalMatrix") -> "BlockDiagonalMatrix":
        """       
        This method performs dot product between corresponding blocks
        of the two block diagonal matrices. The result is a new BlockDiagonalMatrix where
        each block is the element-wise product of the corresponding blocks in the input matrices.
        It overloads the `*` operator.

        :param other: Another BlockDiagonalMatrix to multiply.
        :return: The product as a new BlockDiagonalMatrix.
        """
        assert self.n == other.n and self.d == other.d, "Matrices must have the same dimensions."
        result_blocks = self.blocks * other.blocks
        return BlockDiagonalMatrix(result_blocks)

    def __matmul__(self, other: "BlockDiagonalMatrix") -> "BlockDiagonalMatrix":
        """
        This method performs matrix multiplication by leveraging broadcasting and vectorized operations.
        It computes the result by multiplying corresponding blocks and summing over the common dimension.
        The resulting BlockDiagonalMatrix contains the product of the two input matrices in block diagonal form.
        It overloads the `@` operator.

        :param other: Another BlockDiagonalMatrix to multiply.
        :return: The product as a new BlockDiagonalMatrix.
        """
        assert self.n == other.n and self.d == other.d, "Matrices must have the same dimensions."

        # Expand dimensions to enable broadcasting
        left_blocks = self.blocks[:, :, np.newaxis, :]
        right_blocks = other.blocks[np.newaxis, :, :, :]

        # Element-wise multiplication and sum along the k-axis
        result_blocks = np.sum(left_blocks * right_blocks, axis=1)

        return BlockDiagonalMatrix(result_blocks)

    def inverse(self) -> "BlockDiagonalMatrix":
        """
        Efficiently compute the inverse of a Block Diagonal Matrix in divide and conquer method.
        In case of a singular matrix, it raises an exception.
        
        :return: The inverse as a new BlockDiagonalMatrix.
        """
        inverse_blocks = np.zeros(self.blocks.shape)
        
        for i in range(self.d):
            sub_arr = self.blocks[:, :, i]
            try:
                sub_arr_inv = np.linalg.inv(sub_arr)
            except np.linalg.LinAlgError as e:
                raise Exception("Matrix is singular and cannot be inverted.")
            inverse_blocks[:, :, i] = sub_arr_inv
        
        return BlockDiagonalMatrix(inverse_blocks)

# Recommender-System-Using-SGD

* use SGD algorithm to predict the rating given User Id, and Movie Id.

    
    
    ![equation](https://latex.codecogs.com/svg.image?L%20=%20%5Cmin_%7B%20b,%20c,%20%5C%7B%20u_i%20%5C%7D_%7Bi=1%7D%5EN,%20%5C%7B%20v_j%20%5C%7D_%7Bj=1%7D%5EM%7D%5Cquad%5Calpha%20%5CBig(%20%20%20%20%5Csum_%7Bj%7D%20%5Csum_%7Bk%7D%20v_%7Bjk%7D%5E2%20%20%20%20&plus;%20%5Csum_%7Bi%7D%20%5Csum_%7Bk%7D%20u_%7Bik%7D%5E2%20%20%20%20&plus;%20%5Csum_%7Bi%7D%20b_i%5E2%20%20%20%20&plus;%20%5Csum_%7Bj%7D%20c_i%5E2%20%20%20%20%5CBig)&plus;%20%5Csum_%7Bi,j%20%5Cin%20%5Cmathcal%7BI%7D%5E%7B%5Ctext%7Btrain%7D%7D%7D%20%20%20%20(y_%7Bij%7D%20-%20%5Cmu%20-%20b_i%20-%20c_j%20-%20u_i%5ET%20v_j)%5E2)
    
    
<ul>
<li><span class="math">$\mu$</span> : scalar mean rating</li>
<li><span class="math">$b_i$</span> : scalar bias term for user <span class="math">$i$</span></li>
<li><span class="math">$c_j$</span> : scalar bias term for movie <span class="math">$j$</span></li>
<li><span class="math">$u_i$</span> : K-dimensional vector for user <span class="math">$i$</span></li>
<li><span class="math">$v_j$</span> : K-dimensional vector for movie <span class="math">$j$</span></li>
</ul>


## Formulate as a Graph Problem

* We can construct a **weighted undirect graph** from a given pair of user and movie, and weight will be rating given to movie by user.
* we can construct this matrix like $A[i][j]=r_{ij}$ here $i$ is user_id, $j$ is movie_id and $r_{ij}$ is rating given by user $i$ to the movie $j$


## Implicit User Feautures and Movie Features using Matrix Factorization

We will Apply SVD decomposition on the Adjaceny matrix  $A$ and get three matrices $U, \sum, V$ such that $U \times \sum \times V^T = A$, <br> 
if $A$ is of dimensions $N \times M$ then <br>
$U$ is of $N \times k$, <br>
$\sum$ is of $k \times k$ and <br>
$V$ is $M \times k$ dimensions. <br>

   * Matrix $U$ can be represented as matrix **representation of users**, where each row $u_{i}$ represents a **k-dimensional vector for a user**

   * Matrix $V$ can be represented as matrix **representation of movies**, where each row $v_{j}$ represents a **k-dimensional vector for a movie.**


### Dertivative of Loss Function 
![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%20dL/db_i%20=%20%5Calpha%20*2%20*%20b_i&plus;%20%5CBig(-2*%5Csum_%7Bi,j%20%5Cin%20%5Cmathcal%7BI%7D%5E%7B%5Ctext%7Btrain%7D%7D%7D%20%20%20%20(y_%7Bij%7D%20-%20%5Cmu%20-%20b_i%20-%20c_j%20-%20u_i%5ET%20v_j)%5CBig))

![equation](https://latex.codecogs.com/svg.image?dL/dc_j%20=%5Calpha%20*2%20*%20c_j&plus;%20%5CBig(-2%20*%5Csum_%7Bi,j%20%5Cin%20%5Cmathcal%7BI%7D%5E%7B%5Ctext%7Btrain%7D%7D%7D%20%20%20%20(y_%7Bij%7D%20-%20%5Cmu%20-%20b_i%20-%20c_j%20-%20u_i%5ET%20v_j)%5CBig))


### Training Optimizer

<pre>
for each epoch:
    for each pair of (user, movie):
        b_i =  b_i - learning_rate * dL/db_i
        c_j =  c_j - learning_rate * dL/dc_j
predict the ratings with formula
</pre>
$\hat{y}_{ij} = \mu + b_i + c_j + \text{dot_product}(u_i , v_j) $



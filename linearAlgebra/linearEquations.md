# LINEAR EQUATIONS

y = mx + c

a linear equation has two variables x and y.

m is the gradient.

c tells us about the x intercept.

Equations of the first order are called linear equations

the general form of linear equation is ax + by + c = 0

the more the variables the more planes an equation can make.

this can be extended to n-dimensions.

When more than one linear equation is represented in n-dimensional space, it’s interesting to analyse common points or intersection points of these hyperplanes represented by these equations. These common points lie on all the hyperplanes simultaneously and are known as solutions of the system of
linear equations.

## EXAMPLE USING LINEAR EQUATION

Consider a situation where a group of friends plan to visit a shopping mall. They plan to spend time on movie, bowling, and play station. With difference of opinion on where to start, they get divided into three groups. After spending time in the mall, they all gather at one place for discussion.

1. this is what the first group spent:

   1. 1 bowling alleys
   2. 1 play stations
   3. 1 movie ticket

   - they spent a total of 1500 ksh

2. this is what the second group spent:

   1. 3 bowling alleys
   2. 4 play stations
   3. 2 movie tickets

   - they spent a total of 4400 ksh

3. this is what the third group spent:

   1. 5 bowling alleys
   2. 3 play stations
   3. 5 movie tickets

   - they spent a total of 6500

with this information you need to come up with the cost of a bowling alley, play station and even the movie ticket.

you can use the x axis for bowling alley, y axis for play station and z axis for the movie ticket.

to do this you are going to plot this equations:

1. `1b + 1ps + 1mt = 1500`
2. `3b + 4ps + 2mt = 4400`
3. `5b + 3ps + 5mt = 6500`

where these three graphs are going to meet will give you the price of bowling, the price of play station and the price of the movie ticket, remember that our equation has the x, y and z.

## inconcistent and consistent systems

if a system intersects at only one place, they are called consitent system

if the graphs intersects in a straight line then such a system is called inconcistent system

## SOLVING SYSTEMS OF EQUATION ANALYTICALLY

you don't always need to plot the graphs to see their point of intersection. It will be messy, consider a equation with more than four variables, how are you going to plot that?

there are various ways apart from plotting and they include:

1. Multiplication
2. Addition

### Multiplication

if you multiply the left hand side of the equation and the right hand side of the equation with the same non-zero real number, it will not alter the equality of the equation

3(`1b + 1ps + 1mt`) = 3(`1500`)

### Addition

adding the same real number on the left-hand side (LHS) and the right-hand side (RHS) of the equation doesn't alter the equality of the equation.

3 + `(1b + 1ps + 1mt)` = 3 + (`1500`)

you can also add two LHS's and two RHS's and the equation will still be equal and unaltered.

let's add two equations together:

1. `1b + 1ps + 1mt = 1500`
2. `3b + 4ps + 2mt = 4400`

so you can `(1b + 1ps + 1mt) + (3b + 4ps + 2mt) = (1500 + 4400)`

through a mixture of addition and multiplication you can find the cost of bowling, play stations and movie tickets.

![the solution](../images/linearequation1.jpg)

the above solution shows how you can use the multiplication and addition to solve your linear equations

you fundamentally need at least three equations to solve three unknowns.

if two equations can be obtained from another, that will result to infinitely many solutions.

again there are many equations where one can plot the planes but they never intersect at all, these equations can form parallel planes or planes where at no point do they intersect and this are known as `inconsistent systems`

## MATRICES

as number of variables increase, it becomes to hard to solve this equations manually or using addition and multiplication or using planes.

we also need to automate this equations since i mean, that's what we want right? this leads us to `matrices`. ladies and gentle men, here comes `Uncle Matrix`

### [MATRIX](./matrix.md)

matrix is a rectangular array of numbers for which operations such as addition and multiplication can be defined.

this horizontal and vertical lines of entries in a matrix are called rows and columns. we normally refer to matrices as `m * n matrix`

Each entry is indexed with row and column numbers i.e X(mn) where m rep row and n rep column.

A matrix with the same number of rows and columns; m = n is called a square matrix, represented as Am. A matrix whose entries are only real numbers is called real matrix.

#include <iostream>
#include <cassert>
#include <vector>
#include <functional>
#include <cmath>
#include <fstream>

using namespace std;

//This will contain all the elements needed to interpolate a function
class InterpolatingPolynomial{
    private: 
        //Generate a grid of points from a to b of size h
        vector<float> get_grid(float a, float b, float h) {
            if (a >= b) throw std::invalid_argument("a must be less than b");
            if (h <= 0) throw std::invalid_argument("h must be positive");
            if (h > (b - a)) throw std::invalid_argument("h is larger than the bounds");

            vector<float> grid; 
        
            for (float x = a; x <= b + 1e-6; x += h) { // Add a small epsilon to account for floating-point inaccuracies
                grid.push_back(x);
            }
            return grid;
        }
        //Evaluate a lagrange basis function at x
        float lagrange_basis(float x, int j) {
            float p=1;
            for (int i = 0; i < grid_points.size(); i++){
                if (i==j){
                    p*= 1;
                }
                else{
                    p *= (x-grid_points[i])/(grid_points[j]-grid_points[i]);
                }
            }
            return p;
        }
        //Evaluate the lagrange polynomial (interpolating polynomial) at x
        float lagrange_polynomial(float x) {
            float s = 0;
            for (int i = 0; i < grid_points.size(); i++){
                s+= this->interp_points[i]*lagrange_basis(x,i);
            }
            return s;
        }

    public:
        function<float(float)> func_to_interp; 
        vector<float> grid_points;
        vector<float> interp_points;
        float a;
        float b;
        float h;
        //We just need to pass in the range, step size, and the function to interpolate
        InterpolatingPolynomial(float x, float y, float z, function<float(float)> func) {
            if (x >= y) throw std::invalid_argument("a must be less than b");
            if (z <= 0) throw std::invalid_argument("h must be positive");
            if (z > (y - x)) throw std::invalid_argument("h is larger than the bounds");

            this->a=x;
            this->b=y;
            this->h=z;
            this->grid_points = get_grid(a,b,h); 
            this->interp_points = this->grid_points;
            this->func_to_interp = func; 
            
            //this->interp_points = grid_points; // This will be filled with the calculated y values
            for(int i = 0; i < grid_points.size(); i++){
                interp_points[i] = func(grid_points[i]);
            }
        }
        //Here we'll write pairs (x,y) to a file and plot them using GNUplot. 
        //The datafiles only exist while this method is running, just long enough 
        //so that we can pass it to gnuplot. Im not sure if this is the best way to do
        //this? If there is a way to pass them to gnuplot through memory it would avoid
        // the write to disk.
        void plot(){
                vector<float> x, y1, y2; 
                for (float i = this->a; i < this->b+1e-6; i += 0.01) {
                    x.push_back(i);
                    y1.push_back(this->func_to_interp(i));
                    y2.push_back(this->lagrange_polynomial(i));
                }

                // Write data to a file. care for overflow on i. 
                ofstream dataFile1("data1.txt");
                for (int i = 0; i < x.size(); i++) {
                    dataFile1 << x[i] << " " << y1[i] << endl;
                }
                dataFile1.close();

                // Write data to a file
                ofstream dataFile2("data2.txt");
                for (int i = 0; i < x.size(); i++) {
                    dataFile2 << x[i] << " " << y2[i] << endl;
                }
                dataFile2.close();

                // open a gnuplot pipe that we will write to.
                // it's type is a FILE pointer
                FILE* gnuplotPipe = popen("gnuplot", "w");
                if (gnuplotPipe) {
                    fprintf(gnuplotPipe, "plot 'data1.txt' with lines title 'f(x)', 'data2.txt' with lines title 'Interpolating Polynomial'\n");
                    fflush(gnuplotPipe);
                    cout << "Press enter to close the plot." << endl;
                    cin.get();
                    pclose(gnuplotPipe); //close pipe
                } else {
                    cerr << "Please install GNUPlot." << endl;
                }
                //After it's done, remove the datafiles generated 
                remove("data1.txt");
                remove("data2.txt");
        }
};
//Runge function for part 2
float rung(float x) {
    return 1.0f/(1+x*x);
}
int main()
{
    //1
    float a1 = 0.0f, b1 = 2.0f, h1 = 0.25f;

    function<float(float)> sin_func = static_cast<float(*)(float)>(sin);
    // Print the grid
    InterpolatingPolynomial interp_poly(a1,b1,h1, sin_func);
    
    interp_poly.plot();

    cout <<"Press enter to see the next plot" << endl;
    cin.get();
    //2 
    float a2 = -5.0f, b2 = 5.0f, h2 = 1.0f;

    // Print the grid
    InterpolatingPolynomial interp_poly2(a2,b2,h2, rung);
    
    interp_poly2.plot();

    cout <<"Press enter to see the next plot" << endl;
    cin.get();
    //3 
    float a3 = 0.0f, b3 = 1.0f, h3 = 0.1f;

    function<float(float)> sqrt_func = static_cast<float(*)(float)>(sqrt);
    // Print the grid
    InterpolatingPolynomial interp_poly3(a3,b3,h3, sqrt_func);
    
    interp_poly3.plot();
    cout <<"Press any key to finish the program. Thanks for looking :)";
    cin.get();
    return 0;
}


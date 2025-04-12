// 16-bit XLNS for CUDA
// copyright 1999-2025 Mark G. Arnold
// these routines 
//    ran on 16-bit Turbo C/C++ (the file may have CR/LFs from that system) 
//    were used in my PhD research and for several later papers on 32-bit LNS
// they were ported to Linux gcc and g++ around 2015 on 32-bit x86
// they were ported again for 64-bit arch in 2025, 
// they were copied and modified as __device__ CUDA routines
// they are based on similar math foundation (Gaussian logs, sb and db) as Python xlns,
//    but use different internal storage format:
//    +------+-------------------------+
//    | sign | int(log2) . frac(log2)  |
//    +------+-------------------------+
//    the int(log2) is not twos complement; it is offset (logsignmask XORed)
//    for the 32-bit format in this file, this is roughly similar to float32
//    1 sign bit, 8 int(log2) bits, 23 frac(log2) bits
//    there is an exact representation of 0.0, but no subnormals or NaNs


/* PORTABLE CODE STARTS HERE*/

inline __device__ xlns32 xlns32d_overflow(xlns32 xlns32_x, xlns32 xlns32_y, xlns32 xlns32_temp)
{       
	if (xlns32_logsignmask&xlns32_temp)
		return (xlns32_signmask&(xlns32_x^xlns32_y));
	else
		return (xlns32_signmask&(xlns32_x^xlns32_y))| xlns32_logmask;
}

inline __device__ xlns32 xlns32d_mul(xlns32 x, xlns32 y)
{
   xlns32 xlns32_temp;
   xlns32_temp=(xlns32_logmask&(x))+(xlns32_logmask&(y))-xlns32_logsignmask; 
   return  (xlns32_signmask&(xlns32_temp)) ? xlns32d_overflow(x,y,xlns32_temp) 
                                       :(xlns32_signmask&(x^y))|xlns32_temp;
}

inline __device__ xlns32 xlns32d_div(xlns32 x, xlns32 y)
{
   xlns32 xlns32_temp;
   xlns32_temp=(xlns32_logmask&(x))-(xlns32_logmask&(y))+xlns32_logsignmask; 
   return  (xlns32_signmask&(xlns32_temp)) ? xlns32d_overflow(x,y,xlns32_temp) 
                                       :(xlns32_signmask&(x^y))|xlns32_temp;
}

#ifdef xlns32_ideal
  #define xlns32d_sb xlns32d_sb_ideal
  #define xlns32d_db xlns32d_db_ideal
  #include <math.h>
  inline __device__ xlns32 xlns32d_sb_ideal(xlns32_signed z)
  {
	return ((xlns32) ((log(1+ pow(2.0, ((double) z) / xlns32_scale) )/log(2.0))*xlns32_scale+.5));
  }
  inline __device__ xlns32 xlns32d_db_ideal(xlns32_signed z)
  {
	return ((xlns32_signed) ((log( pow(2.0, ((double) z) / xlns32_scale) - 1 )/log(2.0))*xlns32_scale+.5));
  }
//#else
// no implementation on CUDA for non-ideal yet
#endif



#ifdef xlns32_alt

__device__ inline xlns32 xlns32d_add(xlns32 x, xlns32 y)
{
    xlns32 minxyl, maxxy, xl, yl, usedb, adjust, adjustez;
    xlns32_signed z;

    xl = x & xlns32_logmask;

    yl = y & xlns32_logmask;
    minxyl = (yl>xl) ? xl : yl;
    maxxy  = (xl>yl) ? x  : y;
    z = minxyl - (maxxy&xlns32_logmask);
    usedb = xlns32_signmask&(x^y);

    #ifdef xlns32_ideal

     float pm1 = usedb ? -1.0 : 1.0;

     adjust = z+((xlns32_signed)(log(pm1+pow(2.0,-((double)z)/xlns32_scale))/log(2.0)*xlns32_scale+.5));

    //#else
    // no implementation on CUDA for non-ideal yet
    #endif 
    adjustez = (z < -xlns32_esszer) ? 0 : adjust;
    return ((z==0) && usedb) ?
                     xlns32_zero :
                      xlns32d_mul(maxxy, xlns32_logsignmask + adjustez);
}

#else



//++++ X-X ERROR fixed

__device__ xlns32 xlns32d_add(xlns32 x, xlns32 y)
{
	xlns32 t;
	xlns32_signed z;

	z = (x&xlns32_logmask) - (y&xlns32_logmask);
	if (z<0)
	{
		z = -z;
		t = x;
		x = y;
		y = t;
	}
	if (xlns32_signmask&(x^y))
	{
		if (z == 0)
			return xlns32_zero;
		if (z < xlns32_esszer)
			return xlns32_neg(y + xlns32d_db(z));
		else
			return xlns32_neg(y+z);
	}
	else
	{
		return y + xlns32d_sb(z);
	}
}

#endif

#define xlns32d_sub(x,y) xlns32d_add(x,xlns32_neg(y))

/*END OF PORTABLE CODE*/

/*START OF PORTABLE CODE THAT DEPENDS ON <math.h>*/

#include <math.h>

/*END OF PORTABLE CODE THAT DEPENDS ON <math.h>*/


__device__ xlns32 fp2xlns32d(float x)
{
	if (x==0.0)
		return(xlns32_zero);
	else if (x > 0.0)
		return xlns32_abs((xlns32_signed) ((log(x)/log(2.0))*xlns32_scale))
		       ^xlns32_logsignmask;
	else
		return (((xlns32_signed) ((log(fabs(x))/log(2.0))*xlns32_scale))
			  |xlns32_signmask)^xlns32_logsignmask;
}

__device__ float xlns32d2fp(xlns32 x)
{
	if (xlns32_abs(x) == xlns32_zero)
		return (0.0);
	else if (xlns32_sign(x))
		return (float) (-pow(2.0,((double) (((xlns32_signed) (xlns32_abs(x)-xlns32_logsignmask))))
					/((float) xlns32_scale)));
	else {
		return (float) (+pow(2.0,((double) (((xlns32_signed) (xlns32_abs(x)-xlns32_logsignmask))))
					/((float) xlns32_scale)));
	}
}


#include <iostream>

class xlns32d_float {
    xlns32 x;
 public:
    friend __device__ xlns32d_float operator+(xlns32d_float , xlns32d_float );
    friend __device__ xlns32d_float operator+(float, xlns32d_float );
    friend __device__ xlns32d_float operator+(xlns32d_float , float);
    friend __device__ xlns32d_float operator-(xlns32d_float , xlns32d_float );
    friend __device__ xlns32d_float operator-(float, xlns32d_float );
    friend __device__ xlns32d_float operator-(xlns32d_float , float);
    friend __device__ xlns32d_float operator*(xlns32d_float , xlns32d_float );
    friend __device__ xlns32d_float operator*(float, xlns32d_float );
    friend __device__ xlns32d_float operator*(xlns32d_float , float);
    friend __device__ xlns32d_float operator/(xlns32d_float , xlns32d_float );
    friend __device__ xlns32d_float operator/(float, xlns32d_float );
    friend __device__ xlns32d_float operator/(xlns32d_float , float);
    __device__ xlns32d_float operator=(float); //why not friend?
    friend __device__ xlns32 xlns32d_internal(xlns32d_float );
    friend __device__ float xlns32d_2float(xlns32d_float );
    friend __device__ xlns32d_float float2xlns32d_(float);
    //friend std::ostream& operator<<(std::ostream&, xlns32d_float );
    friend __device__ xlns32d_float operator-(xlns32d_float);
    friend __device__ xlns32d_float operator+=(xlns32d_float &, xlns32d_float);
    friend __device__ xlns32d_float operator+=(xlns32d_float &, float);
    friend __device__ xlns32d_float operator-=(xlns32d_float &, xlns32d_float);
    friend __device__ xlns32d_float operator-=(xlns32d_float &, float);
    friend __device__ xlns32d_float operator*=(xlns32d_float &, xlns32d_float);
    friend __device__ xlns32d_float operator*=(xlns32d_float &, float);
    friend __device__ xlns32d_float operator/=(xlns32d_float &, xlns32d_float);
    friend __device__ xlns32d_float operator/=(xlns32d_float &, float);
    friend __device__ xlns32d_float sin(xlns32d_float);
    friend __device__ xlns32d_float cos(xlns32d_float);
    friend __device__ xlns32d_float exp(xlns32d_float);
    friend __device__ xlns32d_float log(xlns32d_float);
    friend __device__ xlns32d_float atan(xlns32d_float);
    friend __device__ xlns32d_float abs(xlns32d_float);
    friend __device__ xlns32d_float sqrt(xlns32d_float);
//    friend xlns32_float operator-(xlns32_float);
    friend __device__ int operator==(xlns32d_float arg1, xlns32d_float arg2)
      {
       return (arg1.x == arg2.x);
      }
    friend __device__ int operator!=(xlns32d_float arg1, xlns32d_float arg2)
      {
       return (arg1.x != arg2.x);
      }
    friend __device__ int operator<=(xlns32d_float arg1, xlns32d_float arg2)
      {
       return (xlns32_canon(arg1.x)<=xlns32_canon(arg2.x));
      }
    friend __device__ int operator>=(xlns32d_float arg1, xlns32d_float arg2)
      {
       return (xlns32_canon(arg1.x)>=xlns32_canon(arg2.x));
      }
    friend __device__ int operator<(xlns32d_float arg1, xlns32d_float arg2)
      {
       return (xlns32_canon(arg1.x)<xlns32_canon(arg2.x));
      }
    friend __device__ int operator>(xlns32d_float arg1, xlns32d_float arg2)
      {
       return (xlns32_canon(arg1.x)>xlns32_canon(arg2.x));
      }
    friend __device__ int operator==(xlns32d_float arg1, float arg2);
    friend __device__ int operator!=(xlns32d_float arg1, float arg2);
    friend __device__ int operator<=(xlns32d_float arg1, float arg2);
    friend __device__ int operator>=(xlns32d_float arg1, float arg2);
    friend __device__ int operator<(xlns32d_float arg1, float arg2);
    friend __device__ int operator>(xlns32d_float arg1, float arg2);
  };




/*access function for internal representation*/

__device__ xlns32 xlns32d_internal(xlns32d_float y) {
    return y.x;
}

__device__ float xlns32d_2float(xlns32d_float y) {
	return xlns32d2fp(y.x);
}


//no caching
__device__ xlns32d_float float2xlns32d_(float y) {
	xlns32d_float z;
	z.x = fp2xlns32d(y);
	return z;
}


/*overload stream output << operator*/

//#include <ostream>
//std::ostream& operator<< (std::ostream& s, xlns32_float  y) {
//    return s << xlns32_2float(y);
//}

__device__ xlns32d_float operator-(xlns32d_float arg1) {
   xlns32d_float z;
   z.x=xlns32_neg(arg1.x);
   return z;
}



__device__ xlns32d_float operator+(xlns32d_float arg1, xlns32d_float arg2) {
   xlns32d_float z;
   z.x=xlns32d_add(arg1.x,arg2.x);
   return z;
}

__device__ xlns32d_float operator-(xlns32d_float arg1, xlns32d_float arg2) {
   xlns32d_float z;
   z.x=xlns32d_sub(arg1.x,arg2.x);
   return z;
}

__device__ xlns32d_float operator*(xlns32d_float arg1, xlns32d_float arg2) {
   xlns32d_float z;
   z.x=xlns32d_mul(arg1.x,arg2.x);
   return z;
}

__device__ xlns32d_float operator/(xlns32d_float arg1, xlns32d_float arg2) {
   xlns32d_float z;
   z.x=xlns32d_div(arg1.x,arg2.x);
   return z;
}


/*operators with auto type conversion*/

__device__ xlns32d_float operator+(float arg1, xlns32d_float arg2) {
   return float2xlns32d_(arg1)+arg2;
}

__device__ xlns32d_float operator+(xlns32d_float arg1, float arg2) {
   return arg1+float2xlns32d_(arg2);
}


__device__ xlns32d_float operator-(float arg1, xlns32d_float arg2) {
   return float2xlns32d_(arg1)-arg2;
}

__device__ xlns32d_float operator-(xlns32d_float arg1, float arg2) {
   return arg1-float2xlns32d_(arg2);
}

__device__ xlns32d_float operator*(float arg1, xlns32d_float arg2) {
   return float2xlns32d_(arg1)*arg2;
}

__device__ xlns32d_float operator*(xlns32d_float arg1, float arg2) {
   return arg1*float2xlns32d_(arg2);
}


__device__ xlns32d_float operator/(float arg1, xlns32d_float arg2) {
   return float2xlns32d_(arg1)/arg2;
}

__device__ xlns32d_float operator/(xlns32d_float arg1, float arg2) {
   return arg1/float2xlns32d_(arg2);
}

/*comparisons with conversion seems not to inline OK*/

__device__ int operator==(xlns32d_float arg1, float arg2)
      {
       return arg1 == float2xlns32d_(arg2);
      }
__device__ int operator!=(xlns32d_float arg1, float arg2)
      {
       return arg1 != float2xlns32d_(arg2);
      }
__device__ int operator<=(xlns32d_float arg1, float arg2)
      {
       return arg1<=float2xlns32d_(arg2);
      }
__device__ int operator>=(xlns32d_float arg1, float arg2)
      {
       return arg1>=float2xlns32d_(arg2);
      }
__device__ int operator<(xlns32d_float arg1, float arg2)
      {
       return arg1<float2xlns32d_(arg2);
      }
__device__ int operator>(xlns32d_float arg1, float arg2)
      {
       return arg1>float2xlns32d_(arg2);
      }

/*With and without convert:  +=, -=, *=, and /= */

__device__ xlns32d_float operator+=(xlns32d_float & arg1, xlns32d_float arg2) {
   arg1 = arg1+arg2;
   return arg1;
}

__device__ xlns32d_float operator+=(xlns32d_float & arg1, float arg2) {
   arg1 = arg1+float2xlns32d_(arg2);
   return arg1;
}



__device__ xlns32d_float operator-=(xlns32d_float & arg1, xlns32d_float arg2) {
   arg1 = arg1-arg2;
   return arg1;
}

__device__ xlns32d_float operator-=(xlns32d_float & arg1, float arg2) {
   arg1 = arg1-float2xlns32d_(arg2);
   return arg1;
}


__device__ xlns32d_float operator*=(xlns32d_float & arg1, xlns32d_float arg2) {
   arg1 = arg1*arg2;
   return arg1;
}

__device__ xlns32d_float operator*=(xlns32d_float & arg1, float arg2) {
   arg1 = arg1*float2xlns32d_(arg2);
   return arg1;
}


__device__ xlns32d_float operator/=(xlns32d_float & arg1, xlns32d_float arg2) {
   arg1 = arg1/arg2;
   return arg1;
}

__device__ xlns32d_float operator/=(xlns32d_float & arg1, float arg2) {
   arg1 = arg1/float2xlns32d_(arg2);
   return arg1;
}



/*assignment with type conversion*/


__device__ xlns32d_float xlns32d_float::operator=(float rvalue) {
     x = float2xlns32d_(rvalue).x;
   return *this;
}



// functions computed ideally by convert to/from FP


inline __device__ xlns32d_float sin(xlns32d_float x)
{ 
	return float2xlns32d_(sin(xlns32d_2float(x))); 
}

inline __device__ xlns32d_float cos(xlns32d_float x)
{ 
	return float2xlns32d_(cos(xlns32d_2float(x))); 
}

// exp and log can be implemented more efficiently in LNS but 
// this is just cookie cutter ideal implementation at present

inline __device__ xlns32d_float exp(xlns32d_float x)
{ 
	return float2xlns32d_(exp(xlns32d_2float(x))); 
}

inline __device__ xlns32d_float log(xlns32d_float x)
{ 
	return float2xlns32d_(log(xlns32d_2float(x))); 
}

inline __device__ xlns32d_float atan(xlns32d_float x)
{ 
	return float2xlns32d_(atan(xlns32d_2float(x))); 
}

// the following have efficient macro implementations

inline __device__ xlns32d_float sqrt(xlns32d_float x)
{ 
	xlns32d_float result;
	result.x = xlns32_sqrt(x.x); 
	return result; 
}

inline __device__ xlns32d_float abs(xlns32d_float x)
{ 
	xlns32d_float result;
	result.x = xlns32_abs(x.x); 
	return result; 
}


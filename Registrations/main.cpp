#include "registration.h"

using namespace std;

int main() {
    ImageProcessor processor;
    FeatureMatcher matcher(5000);
    AffineTransformer transformer;

    ImageRegistration registration(processor, matcher, transformer);

    registration.processImages("C:\\Users\\pahul\\Pictures\\Pair\\1.jpg",
                               "C:\\Users\\pahul\\Pictures\\Pair\\2.jpg");

    return 0;
}

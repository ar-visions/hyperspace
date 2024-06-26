#pragma once
#include <mx/mx.hpp>

namespace ion {

struct AInternal;

struct AI:mx {
    using intern = AInternal;
    ptr_declare(AI);
    AI(path_t p);
    Array<float> operator()(Array<mx> v);
};

#if 0

struct DataW {
    path_t      p;
    var         data;
    float       w;
};

struct Truth {
    Array<float>  label;
    Array<var>    data;
    ///
    Truth(null_t n = nullptr) { }
    operator bool()  { return data and label.len(); }
    bool operator!() { return !(operator bool()); }
};

typedef Array<Truth> Truths;

struct Dataset {
    path_t      path;
    str         dataset;
    float       w;
    ///
    void import(path_t root, str d) {
        auto sp = d.split(":");
        path    = var::format("{0}/{1}", {root.string(), sp[0]});
        dataset = sp[0];
        w       = sp.size() > 1 ? sp[1].real() : 1.0;
    }
    Dataset(path_t root, str d) {
        import(root, d);
    }
    Dataset(var &d) {
        import(path_t(d[size_t(0)]), str(d[size_t(1)]));
    }
    static Array<Dataset> parse(path_t root, str ds) {
        Array<Dataset> r;
        Array<str>     sp = ds.split(",");
        for (str &d: sp) {
            r += {root, d};
        }
        return r;
    }
};

void Gen(Map            &args,
         Array<str>      require,
         Array<Dataset> &ds,
         str             model,
         lambda<Truths(var &)> fn);

Truths if_image(var &data, type_t format, lambda<Truths(Image &)> fn);
Truths if_audio(var &data, lambda<Truths(Audio &)> fn);

#endif
}

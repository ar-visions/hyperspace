#pragma once

/// two valid meanings inferred
struct AInternal;

struct AI:mx {
protected:
    AInternal *i;
public:
    ptr(AI, mx, AInternal, i);
    AI(path_t p);
    array<float> operator()(array<mx> v);
};

struct DataW {
    path_t      p;
    var         data;
    float       w;
};

struct Truth {
    array<float>  label;
    array<var>    data;
    ///
    Truth(std::nullptr_t n = nullptr) { }
    operator bool()  { return data and label.size(); }
    bool operator!() { return !(operator bool()); }
};

typedef array<Truth> Truths;

struct Dataset {
    path_t      path;
    str         dataset;
    float       w;
    ///
    operator var()  {
        return array<var> { var(path.string()), var(dataset), var(w) };
    }
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
    static array<Dataset> parse(path_t root, str ds) {
        array<Dataset> r;
        array<str>     sp = ds.split(",");
        for (str &d: sp) {
            r += {root, d};
        }
        return r;
    }
};

void Gen(Map            &args,
         array<str>      require,
         array<Dataset> &ds,
         str             model,
         lambda<Truths(var &)> fn);

Truths if_image(var &data, type_t format, lambda<Truths(Image &)> fn);
Truths if_audio(var &data, lambda<Truths(Audio &)> fn);

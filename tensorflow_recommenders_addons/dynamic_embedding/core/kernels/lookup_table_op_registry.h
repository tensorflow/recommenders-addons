#ifndef TFRA_CORE_KERNELS_OP_REGISTRY_H_
#define TFRA_CORE_KERNELS_OP_REGISTRY_H_

#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/lookup_table_interface.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"


namespace tensorflow {
namespace recommenders_addons {
namespace op_registry {

using namespace lookup_table;

class OpKernelTableFactory {
 public:
  virtual LookupTableInterface* Create(OpKernelContext *ctx, OpKernel *kernel) = 0;
  virtual ~OpKernelTableFactory() = default;
}; // class OpKernelTableFactory

class LookupTableOpRegistry {
 private:
   std::map<std::string, std::unique_ptr<OpKernelTableFactory> > DeferRegistrationData_;

 private:
   struct PtrOpKernelTableFactory : public OpKernelTableFactory {
      explicit PtrOpKernelTableFactory(LookupTableInterface* (*create_func)(OpKernelContext *, OpKernel *))
         : create_func_(create_func) {}
      
      LookupTableInterface* Create(OpKernelContext *ctx, OpKernel *kernel) override;

      LookupTableInterface* (*create_func_)(OpKernelContext *, OpKernel *);
   }; // struct PtrOpKernelTableFactory

   LookupTableOpRegistry(){}

 public:
   ~LookupTableOpRegistry(){}

   template<typename K, typename V>
   void DeferRegister(const std::string& op_name, 
                      const std::string& device, 
                      LookupTableInterface* (create_fn)(OpKernelContext *, OpKernel *));

   LookupTableInterface* LookUp(const string &lookup_table_name,
                                OpKernelContext *ctx, OpKernel *kernel);

   // static LookupTableOpRegistry* Global();

   static LookupTableOpRegistry* Global() {
      static op_registry::LookupTableOpRegistry tfra_global_op_registry;
      return &tfra_global_op_registry;
   }
};


#define REGISTER_LOOKUP_TABLE_IMPL_3(ctr, op_name, device, key_type, value_type, ...)                                    \
   static ::tensorflow::InitOnStartupMarker const Klookup_table_##ctr TF_ATTRIBUTE_UNUSED =                              \
         TF_INIT_ON_STARTUP_IF(true)                                                                                     \
         << ([]() {                                                                                                      \
            ::tensorflow::recommenders_addons::op_registry::LookupTableOpRegistry::Global()->                            \
                  DeferRegister<key_type, value_type>(op_name, device,                                                   \
                           [](::tensorflow::OpKernelContext *ctx, ::tensorflow::OpKernel *kernel)                        \
                              -> LookupTableInterface* {                                                                 \
                                 return new __VA_ARGS__(ctx, kernel);                                                    \
                              });                                                                                        \
            return ::tensorflow::InitOnStartupMarker{};                                                                  \
         })();

#define REGISTER_LOOKUP_TABLE_IMPL_2(...)   \
   TF_NEW_ID_FOR_INIT(REGISTER_LOOKUP_TABLE_IMPL_3, __VA_ARGS__)

#define REGISTER_LOOKUP_TABLE_IMPL(...) \
   REGISTER_LOOKUP_TABLE_IMPL_2(__VA_ARGS__)

#define REGISTER_LOOKUP_TABLE(...) \
   REGISTER_LOOKUP_TABLE_IMPL(__VA_ARGS__)

}   // namespace op_registry
}   // namespace recommenders_addons
}   // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_OP_REGISTRY_H_
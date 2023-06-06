#ifndef TFRA_CORE_KERNELS_OP_REGISTRY_H_
#define TFRA_CORE_KERNELS_OP_REGISTRY_H_

#include <map>
#include <memory>
#include <string>
#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/lookup_table_interface.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/lookup_registration.h"
// #include "tensorflow/core/framework/registration/registration.h"
// #include "tensorflow/core/framework/op_kernel.h"
// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/lib/strings/strcat.h"
// #include "tensorflow/core/lib/core/stringpiece.h"
// #include "tensorflow/core/framework/kernel_def.pb.h"
// #include "tensorflow/core/framework/kernel_def_builder.h"


namespace tensorflow {
namespace recommenders_addons {
namespace op_registry {

using namespace lookup_table;

template<class K, class V>
class OpKernelTableFactory {
 public:
//   virtual LookupTableInterface* Create(OpKernelContext *ctx, OpKernel *kernel) = 0;
  virtual TFRALookupTableInterface<K, V>* Create(const KVTableInfo& info) = 0;
  virtual ~OpKernelTableFactory() = default;
}; // class OpKernelTableFactory

template<class K, class V>
class LookupTableOpRegistry {
 private:
   std::map<std::string, std::unique_ptr<OpKernelTableFactory<K, V> > > DeferRegistrationData_;

 private:
   struct PtrOpKernelTableFactory : public OpKernelTableFactory<K, V> {
      // explicit PtrOpKernelTableFactory(LookupTableInterface* (*create_func)(OpKernelContext *, OpKernel *))
      explicit PtrOpKernelTableFactory(TFRALookupTableInterface<K, V>* (*create_func)(const KVTableInfo& info))
         : create_func_(create_func) {}
      
      // LookupTableInterface* Create(OpKernelContext *ctx, OpKernel *kernel) override;
      TFRALookupTableInterface<K, V>* Create(const KVTableInfo& info) override;

      // LookupTableInterface* (*create_func_)(OpKernelContext *, OpKernel *);
      TFRALookupTableInterface<K, V>* (*create_func_)(const KVTableInfo& info);

   }; // struct PtrOpKernelTableFactory

   LookupTableOpRegistry(){}

 public:
   ~LookupTableOpRegistry(){}

   // template<typename K, typename V>
   // void DeferRegister(const std::string& op_name, 
   //                    const std::string& device, 
   //                    LookupTableInterface* (create_fn)(OpKernelContext *, OpKernel *));

   // template<typename K, typename V>
   void DeferRegister(const std::string& op_name, 
                      const std::string& device, 
                      TFRALookupTableInterface<K, V>* (create_fn)(const KVTableInfo& info));

   // LookupTableInterface* LookUp(const string &lookup_table_name,
   //                              OpKernelContext *ctx, OpKernel *kernel);
   
   TFRALookupTableInterface<K, V>* LookUp(const std::string &lookup_table_name, const KVTableInfo& info);

   // static LookupTableOpRegistry* Global();

   static LookupTableOpRegistry<K, V>* Global() {
      static op_registry::LookupTableOpRegistry<K, V> tfra_global_op_registry;
      return &tfra_global_op_registry;
   }
};

// [](::tensorflow::OpKernelContext *ctx, ::tensorflow::OpKernel *kernel)

#define REGISTER_LOOKUP_TABLE_IMPL_3(ctr, op_name, device, key_type, value_type, ...)                                    \
   static InitOnStartupMarker const lookup_table_##ctr TFRA_ATTRIBUTE_UNUSED =                                           \
         TF_INIT_ON_STARTUP_IF(true)                                                                                     \
         << ([]() {                                                                                                      \
            ::tensorflow::recommenders_addons::op_registry::LookupTableOpRegistry<key_type, value_type>::Global()->      \
                  DeferRegister(op_name, device,                                                                         \
                           [](const KVTableInfo& info)                                                                   \
                              -> TFRALookupTableInterface<key_type, value_type>* {                                       \
                                 return new __VA_ARGS__(info);                                                           \
                              });                                                                                        \
            return InitOnStartupMarker{};                                                                                \
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